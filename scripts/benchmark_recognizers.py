"""
benchmark_recognizers.py
------------------------
Benchmarking pipeline for comparing face RECOGNITION models under drone
hardware constraints. All models are pretrained on face recognition datasets
(VGGFace2, MS1MV2, WebFace600K, etc.) — not ImageNet.

Detector (UltraFace) stays fixed. Only the recognizer is swapped.

Supported models:
  1. facenet         — InceptionResnetV1 pretrained on VGGFace2     (~23.5M params, pip)
  2. mobilefacenet   — MobileFaceNet (ArcFace, WebFace600K)         (~1.2M params, auto-download)
  3. arcface_r50     — ArcFace ResNet-50 (WebFace600K)              (~43M params, auto-download)
  4. arcface_r18     — ArcFace IResNet-18 (MS1MV2)                  (~12M params, manual download)

Install:
    pip install facenet-pytorch insightface onnxruntime

Usage:
    python benchmark_recognizers.py --model facenet \\
        --dataset-root "archive (1)" --dataset-type vggface2

    python benchmark_recognizers.py --all --dataset-root open_data_set \\
        --dataset-type droneface

    python benchmark_recognizers.py --all --dataset-root "archive (1)" \\
        --dataset-type vggface2 --constrained
"""

import os
import sys
import argparse
import json
import statistics
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


# ── Model registry ───────────────────────────────────────────────────────────
# Each model returns embeddings for face images. Some are PyTorch, some ONNX.
# We wrap everything so the benchmark engine gets a consistent interface.

EMBEDDING_DIM = 512  # all face recognition models output 512-dim
INPUT_SIZE_DEFAULT = 112


class ModelWrapper(object):
    """Common interface for all models (PyTorch or ONNX)."""

    def __init__(self, name, input_size=112):
        self.name = name
        self.input_size = input_size
        self._param_count = 0
        self._size_mb = 0.0

    def get_embeddings(self, images_tensor):
        """Takes a (B, 3, H, W) tensor, returns (B, 512) tensor of embeddings."""
        raise NotImplementedError

    def param_count(self):
        return self._param_count

    def size_mb(self):
        return self._size_mb


class PyTorchModelWrapper(ModelWrapper):
    """Wraps a PyTorch nn.Module."""

    def __init__(self, name, model, input_size=112):
        super(PyTorchModelWrapper, self).__init__(name, input_size)
        self.model = model.eval()
        self._param_count = sum(p.numel() for p in model.parameters())
        # Measure size
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            temp_path = f.name
        try:
            torch.save(model.state_dict(), temp_path)
            self._size_mb = round(os.path.getsize(temp_path) / (1024 * 1024), 3)
        finally:
            os.unlink(temp_path)

    def get_embeddings(self, images_tensor):
        with torch.no_grad():
            emb = self.model(images_tensor)
        # Some models return tuples
        if isinstance(emb, tuple):
            emb = emb[0]
        return emb


class ONNXModelWrapper(ModelWrapper):
    """Wraps an ONNX model loaded via onnxruntime."""

    def __init__(self, name, onnx_path, input_size=112, param_count=0, use_bgr=True):
        super(ONNXModelWrapper, self).__init__(name, input_size)
        import onnxruntime
        self.session = onnxruntime.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self._size_mb = round(os.path.getsize(onnx_path) / (1024 * 1024), 3)
        self._param_count = param_count
        self.use_bgr = use_bgr

    def get_embeddings(self, images_tensor):
        # Input is RGB [0, 1] from ToTensor(). Convert to model's expected format.
        images_np = images_tensor.numpy()
        images_np = images_np * 255.0
        if self.use_bgr:
            images_np = images_np[:, [2, 1, 0], :, :]  # RGB to BGR (InsightFace)
        # Normalize to [-1, 1]
        images_np = (images_np - 127.5) / 127.5
        images_np = images_np.astype(np.float32)
        outputs = self.session.run(None, {self.input_name: images_np})
        return torch.from_numpy(outputs[0])


# ── Model factories ──────────────────────────────────────────────────────────

def _make_facenet():
    """
    FaceNet InceptionResnetV1 pretrained on VGGFace2.
    ~23.5M params, 512-dim embeddings, input 160x160.
    Weights auto-download via facenet-pytorch.
    """
    from facenet_pytorch import InceptionResnetV1
    model = InceptionResnetV1(pretrained="vggface2").eval()
    return PyTorchModelWrapper("facenet", model, input_size=160)


def _make_inceptionresnet_raw():
    """
    Raw InceptionResnetV1 baseline: VGGFace2-pretrained backbone with NO fine-tuning
    and NO projection head. This is the off-the-shelf 'no-training-of-our-own' baseline
    that any further-trained variant (e.g. our ArcFace fine-tune) is expected to beat.
    Functionally identical to _make_facenet() but registered under a separate, clearly
    named alias for paper-table clarity.
    """
    from facenet_pytorch import InceptionResnetV1
    model = InceptionResnetV1(pretrained="vggface2").eval()
    return PyTorchModelWrapper("inceptionresnet_raw", model, input_size=160)


_INSIGHTFACE_PACKS = {
    "buffalo_sc": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip",
    "buffalo_l":  "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    "antelopev2": "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip",
}


def _ensure_insightface_pack(pack_name):
    """Download + extract an insightface model pack directly from GitHub releases.
    Avoids needing the insightface package (which requires VS build tools on py36)."""
    home = os.path.expanduser("~")
    model_dir = os.path.join(home, ".insightface", "models", pack_name)
    if os.path.isdir(model_dir) and any(f.endswith(".onnx") for f in os.listdir(model_dir)):
        return model_dir

    if pack_name not in _INSIGHTFACE_PACKS:
        raise RuntimeError("Unknown insightface pack: %s" % pack_name)

    os.makedirs(model_dir, exist_ok=True)
    zip_path = os.path.join(model_dir, "%s.zip" % pack_name)
    url = _INSIGHTFACE_PACKS[pack_name]
    print("  [bench] Downloading %s pack from %s ..." % (pack_name, url))
    import urllib.request, zipfile, shutil
    urllib.request.urlretrieve(url, zip_path)
    print("  [bench] Extracting %s ..." % pack_name)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(model_dir)
    os.remove(zip_path)
    # Flatten single nested subdirectory if present (antelopev2/antelopev2/*.onnx)
    nested = os.path.join(model_dir, pack_name)
    if os.path.isdir(nested):
        for fname in os.listdir(nested):
            shutil.move(os.path.join(nested, fname), os.path.join(model_dir, fname))
        os.rmdir(nested)
    return model_dir


def _make_mobilefacenet():
    """
    MobileFaceNet with ArcFace loss, trained on WebFace600K.
    ~1.2M params, 512-dim embeddings, input 112x112.
    Auto-downloads via insightface buffalo_sc model pack.
    """
    model_dir = _ensure_insightface_pack("buffalo_sc")
    onnx_path = os.path.join(model_dir, "w600k_mbf.onnx")
    if not os.path.exists(onnx_path):
        raise RuntimeError("MobileFaceNet ONNX not found at %s" % onnx_path)
    return ONNXModelWrapper("mobilefacenet", onnx_path, input_size=112, param_count=1200000)


def _make_arcface_r50():
    """
    ArcFace ResNet-50, trained on WebFace600K.
    ~43M params, 512-dim embeddings, input 112x112.
    Auto-downloads via insightface buffalo_l model pack.
    """
    model_dir = _ensure_insightface_pack("buffalo_l")
    onnx_path = os.path.join(model_dir, "w600k_r50.onnx")
    if not os.path.exists(onnx_path):
        raise RuntimeError("ArcFace-R50 ONNX not found at %s" % onnx_path)
    return ONNXModelWrapper("arcface_r50", onnx_path, input_size=112, param_count=43600000)


def _make_arcface_r18():
    """
    ArcFace IResNet-18, trained on MS1MV2.
    ~12.4M params, 512-dim embeddings, input 112x112.

    Requires manual weight download. If weights not found, falls back to
    a randomly initialized model (for latency/size benchmarking only).

    Download from the insightface arcface_torch model list:
    https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
    Save as: weights/arcface_r18.pth
    """
    # IResNet-18 architecture (standard for face recognition)
    class BasicBlock(nn.Module):
        expansion = 1
        def __init__(self, inplanes, planes, stride=1, downsample=None):
            super(BasicBlock, self).__init__()
            self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
            self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
            self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
            self.prelu = nn.PReLU(planes)
            self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
            self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
            self.downsample = downsample

        def forward(self, x):
            identity = x
            out = self.bn1(x)
            out = self.conv1(out)
            out = self.bn2(out)
            out = self.prelu(out)
            out = self.conv2(out)
            out = self.bn3(out)
            if self.downsample is not None:
                identity = self.downsample(x)
            return out + identity

    class IResNet(nn.Module):
        def __init__(self, block, layers, num_features=512):
            super(IResNet, self).__init__()
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            self.bn1 = nn.BatchNorm2d(64, eps=1e-05)
            self.prelu = nn.PReLU(64)
            self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
            self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05)
            self.fc = nn.Linear(512 * block.expansion * 7 * 7, num_features)
            self.features = nn.BatchNorm1d(num_features, eps=1e-05)
            nn.init.constant_(self.features.weight, 1.0)
            self.features.weight.requires_grad = False

        def _make_layer(self, block, planes, blocks, stride=1):
            downsample = None
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
                )
            layers = [block(self.inplanes, planes, stride, downsample)]
            self.inplanes = planes * block.expansion
            for _ in range(1, blocks):
                layers.append(block(self.inplanes, planes))
            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.prelu(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.bn2(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            x = self.features(x)
            return x

    model = IResNet(BasicBlock, [2, 2, 2, 2], 512)

    # Try to load pretrained weights
    weight_paths = [
        "weights/arcface_r18.pth",
        "arcface_r18.pth",
        "ms1mv3_arcface_r18.pth",
    ]
    loaded = False
    for wp in weight_paths:
        if os.path.exists(wp):
            print("  [bench] Loading ArcFace-R18 weights from %s" % wp)
            state = torch.load(wp, map_location="cpu")
            # Handle different checkpoint formats
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
            loaded = True
            break

    if not loaded:
        print("  [bench] WARNING: No pretrained weights found for ArcFace-R18.")
        print("  [bench] Running with random weights (latency/size are valid,")
        print("  [bench] accuracy will be meaningless).")
        print("  [bench] Download weights and save as weights/arcface_r18.pth")

    model.eval()
    return PyTorchModelWrapper("arcface_r18", model, input_size=112)


def _make_facenet_casia():
    """
    FaceNet InceptionResnetV1 pretrained on CASIA-WebFace.
    Same architecture as facenet but different training dataset.
    ~23.5M params, 512-dim embeddings, input 160x160.
    """
    from facenet_pytorch import InceptionResnetV1
    model = InceptionResnetV1(pretrained="casia-webface").eval()
    return PyTorchModelWrapper("facenet_casia", model, input_size=160)


def _make_arcface_r100():
    """
    ArcFace GlintR100 (ResNet-100 variant), trained on Glint360K.
    ~65M params, 512-dim embeddings, input 112x112.
    Auto-downloads via insightface antelopev2 model pack.
    Heaviest model — serves as upper-bound reference.
    """
    model_dir = _ensure_insightface_pack("antelopev2")
    onnx_path = os.path.join(model_dir, "glintr100.onnx")
    if not os.path.exists(onnx_path):
        raise RuntimeError("GlintR100 ONNX not found at %s" % onnx_path)
    return ONNXModelWrapper("arcface_r100", onnx_path, input_size=112, param_count=65000000)


def _download_file(url, dest_path):
    """Download a file from a URL if it doesn't already exist."""
    if os.path.exists(dest_path):
        return dest_path
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print("  [bench] Downloading %s ..." % os.path.basename(dest_path))
    import urllib.request
    urllib.request.urlretrieve(url, dest_path)
    print("  [bench] Saved to %s" % dest_path)
    return dest_path


def _make_sface():
    """
    SFace (Sigmoid-Constrained Hypersphere Loss), from OpenCV Model Zoo.
    ~1.1M params, 128-dim embeddings, input 112x112.
    Very lightweight — auto-downloads ONNX from GitHub.
    Expects RGB input (no BGR swap).
    """
    onnx_path = os.path.join("weights", "face_recognition_sface_2021dec.onnx")
    url = ("https://github.com/opencv/opencv_zoo/raw/main/"
           "models/face_recognition_sface/face_recognition_sface_2021dec.onnx")
    _download_file(url, onnx_path)
    return ONNXModelWrapper("sface", onnx_path, input_size=112,
                            param_count=1100000, use_bgr=False)


class LBPHWrapper(ModelWrapper):
    """
    Classical LBPH (Local Binary Pattern Histogram) face recognizer from OpenCV.
    Produces a concatenated histogram feature vector per image, which we treat
    as an embedding and feed into the same cosine/nearest-centroid evaluator as
    the deep models. This keeps the protocol consistent across all recognizers.

    No "parameters" in the neural sense — we report a conventional 0 for
    param_count and 0.0 MB for size (LBPH has no learned weights).
    """

    def __init__(self, radius=1, neighbors=8, grid_x=8, grid_y=8, input_size=112):
        super(LBPHWrapper, self).__init__("lbph", input_size)
        import cv2
        self._cv2 = cv2
        self.radius = radius
        self.neighbors = neighbors
        self.grid_x = grid_x
        self.grid_y = grid_y
        self._param_count = 0
        self._size_mb = 0.0

    def get_embeddings(self, images_tensor):
        import numpy as _np
        import torch as _torch
        cv2 = self._cv2

        # images_tensor: (B, 3, H, W) RGB float [0, 1]
        B = images_tensor.size(0)
        imgs_np = (images_tensor.numpy() * 255.0).astype(_np.uint8)
        # (B, 3, H, W) -> list of (H, W) grayscale
        gray_imgs = []
        for i in range(B):
            rgb = imgs_np[i].transpose(1, 2, 0)  # (H, W, 3) RGB
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            gray_imgs.append(gray)

        # Train a fresh LBPH recognizer on this batch; getHistograms() returns
        # one histogram per input image, in the same order.
        rec = cv2.face.LBPHFaceRecognizer_create(
            radius=self.radius,
            neighbors=self.neighbors,
            grid_x=self.grid_x,
            grid_y=self.grid_y,
        )
        labels = _np.arange(B, dtype=_np.int32)
        rec.train(gray_imgs, labels)
        hists = rec.getHistograms()  # list of (1, D) float32 arrays

        feats = _np.stack([h.flatten() for h in hists], axis=0).astype(_np.float32)
        return _torch.from_numpy(feats)


def _make_lbph():
    """
    Local Binary Pattern Histogram recognizer (OpenCV classical baseline).
    No learned weights. Grayscale input at 112x112; feature dim ~16,384
    (64 grid cells x 256 bins default). Serves as a classical-method baseline
    against the deep recognizers.
    """
    # Sanity check: the face module must be present (opencv-contrib-python).
    import cv2
    if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        raise RuntimeError(
            "cv2.face.LBPHFaceRecognizer_create not available. "
            "Install opencv-contrib-python (same version as opencv-python)."
        )
    return LBPHWrapper(input_size=112)


def _make_mobilefacenet_int8():
    """
    INT8-quantized MobileFaceNet (QDQ ONNX format).
    Same architecture and contract as the FP32 MobileFaceNet — float32 input,
    512-dim output — but the weights are quantized to INT8 and quant/dequant
    happens in-graph. Expected to run faster and smaller than FP32 with some
    accuracy trade.
    """
    onnx_path = os.path.join("weights", "mobilefacenet_int8.onnx")
    if not os.path.exists(onnx_path):
        raise RuntimeError("INT8 MobileFaceNet ONNX not found at %s" % onnx_path)
    return ONNXModelWrapper("mobilefacenet_int8", onnx_path, input_size=112,
                            param_count=1200000)


MODEL_REGISTRY = {
    "facenet":              (_make_facenet,              "FaceNet InceptionResnetV1 (VGGFace2, ~23.5M params)"),
    "facenet_casia":        (_make_facenet_casia,        "FaceNet InceptionResnetV1 (CASIA-WebFace, ~23.5M params)"),
    "inceptionresnet_raw":  (_make_inceptionresnet_raw,  "Raw InceptionResnetV1 baseline (VGGFace2-pretrained, no fine-tuning, no projection head)"),
    "mobilefacenet":        (_make_mobilefacenet,        "MobileFaceNet ArcFace (WebFace600K, ~1.2M params)"),
    "mobilefacenet_int8":   (_make_mobilefacenet_int8,   "MobileFaceNet INT8-quantized (~1.2M params, ~3.4 MB)"),
    "arcface_r18":          (_make_arcface_r18,          "ArcFace IResNet-18 (~24M params)"),
    "arcface_r50":          (_make_arcface_r50,          "ArcFace ResNet-50 (WebFace600K, ~43M params)"),
    "arcface_r100":         (_make_arcface_r100,         "ArcFace GlintR100 (Glint360K, ~65M params)"),
    "sface":                (_make_sface,                "SFace (OpenCV Zoo, ~1.1M params)"),
    "lbph":                 (_make_lbph,                 "LBPH classical (OpenCV, no learned weights)"),
}


# ── Dataset loaders ──────────────────────────────────────────────────────────

def make_transform(input_size):
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])


class VGGFace2Dataset(Dataset):
    """Loads VGG Face2 from directory structure: root/split/identity/image.jpg"""

    def __init__(self, root, split="val", input_size=112):
        self.root = Path(root) / split
        self.transform = make_transform(input_size)
        self.classes = sorted([
            d.name for d in self.root.iterdir() if d.is_dir()
        ])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.samples = []
        for cls_name in self.classes:
            cls_dir = self.root / cls_name
            for img_path in sorted(cls_dir.iterdir()):
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((str(img_path), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


class DroneFaceDataset(Dataset):
    """
    Loads DroneFace from photos_all_faces folder.
    Parses filename metadata: subject, camera, height, distance.
    """

    HEIGHT_MAP = {"0": 1.5, "3": 3.0, "4": 4.0, "5": 5.0, "na": None}

    def __init__(self, root, input_size=112):
        self.root = Path(root) / "photos_all_faces"
        self.transform = make_transform(input_size)
        self.subjects = sorted(set(
            f.name.split("_")[0] for f in self.root.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ))
        self.class_to_idx = {s: i for i, s in enumerate(self.subjects)}
        self.classes = self.subjects

        self.samples = []
        self.metadata = []

        for img_path in sorted(self.root.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            parts = img_path.stem.split("_")
            subject_id = parts[0] if len(parts) > 0 else "unknown"
            camera = parts[1] if len(parts) > 1 else "na"
            height_id = parts[2] if len(parts) > 2 else "na"
            distance_id = parts[4] if len(parts) > 4 else "na"

            label = self.class_to_idx.get(subject_id, -1)
            if label < 0:
                continue

            height_m = self.HEIGHT_MAP.get(height_id, None)
            try:
                distance_m = 17.0 - (int(distance_id) / 2.0)
            except (ValueError, TypeError):
                distance_m = None

            self.samples.append((str(img_path), label))
            self.metadata.append({
                "subject": subject_id,
                "camera": camera,
                "height_m": height_m,
                "distance_m": distance_m,
                "height_id": height_id,
                "distance_id": distance_id,
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

    def get_metadata(self, idx):
        return self.metadata[idx]


# ── Benchmark engine ─────────────────────────────────────────────────────────

def benchmark_model(wrapped_model, dataset, batch_size=1, constrained=False,
                    save_embeddings_dir=None, dataset_name=None):
    """
    Run a recognition model through the dataset in eval mode.
    Works with both PyTorch and ONNX models via ModelWrapper.

    If constrained=True, model inference runs inside DroneInferenceContext
    (400MHz single core, 128MB RAM delta). Data loading runs at full PC speed.

    If save_embeddings_dir is given, dumps:
      <dir>/<model>_<dataset>.npy            embeddings (N, D)
      <dir>/<model>_<dataset>_labels.npy     labels     (N,)
      <dir>/<model>_<dataset>_filenames.csv  one path per line
    """
    from drone_constraints import DroneInferenceContext
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0)

    # Phase 1: Extract all embeddings + measure latency
    all_embeddings = []
    all_labels = []
    latencies_ms = []
    peak_ram_mb = 0.0
    peak_delta_ram_mb = 0.0

    print("  [bench] Extracting embeddings (%d images)..." % len(dataset))
    for batch_idx, (images, labels) in enumerate(loader):
        # images loaded at full PC speed — constraint only wraps inference
        if constrained:
            ctx = DroneInferenceContext()
            with ctx:
                emb = wrapped_model.get_embeddings(images)
            elapsed_ms = ctx.latency_ms
            if ctx.peak_ram_mb > peak_ram_mb:
                peak_ram_mb = ctx.peak_ram_mb
            if ctx.delta_ram_mb > peak_delta_ram_mb:
                peak_delta_ram_mb = ctx.delta_ram_mb
        else:
            t0 = time.perf_counter()
            emb = wrapped_model.get_embeddings(images)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

        latencies_ms.append(elapsed_ms / images.size(0))

        # Ensure tensor output
        if not isinstance(emb, torch.Tensor):
            emb = torch.from_numpy(emb)
        all_embeddings.append(emb.cpu().float())
        all_labels.append(labels)

        # Progress every 2000 images
        if (batch_idx + 1) % 2000 == 0:
            print("    %d / %d images done..." % (batch_idx + 1, len(dataset)))

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Optional: persist embeddings for downstream analysis
    if save_embeddings_dir is not None:
        os.makedirs(save_embeddings_dir, exist_ok=True)
        ds_tag = dataset_name or "dataset"
        emb_path = os.path.join(save_embeddings_dir, "%s_%s.npy" % (wrapped_model.name, ds_tag))
        lbl_path = os.path.join(save_embeddings_dir, "%s_%s_labels.npy" % (wrapped_model.name, ds_tag))
        fn_path  = os.path.join(save_embeddings_dir, "%s_%s_filenames.csv" % (wrapped_model.name, ds_tag))
        np.save(emb_path, all_embeddings.numpy())
        np.save(lbl_path, all_labels.numpy())
        with open(fn_path, "w") as fh:
            fh.write("idx,filename,label\n")
            for i in range(len(dataset)):
                path, lbl = dataset.samples[i]
                fh.write("%d,%s,%d\n" % (i, path, lbl))
        print("  [bench] Saved embeddings to %s (and _labels.npy / _filenames.csv)" % emb_path)

    # Phase 2: Nearest-centroid accuracy (leave-one-out)
    print("  [bench] Computing accuracy (nearest-centroid)...")
    num_classes = len(dataset.classes)
    total = all_embeddings.size(0)

    # Precompute per-class sums and counts
    class_sums = torch.zeros(num_classes, all_embeddings.size(1))
    class_counts = torch.zeros(num_classes)
    for i in range(total):
        lbl = all_labels[i].item()
        class_sums[lbl] += all_embeddings[i]
        class_counts[lbl] += 1

    # Vectorized accuracy computation (Rank-1 and Rank-5 leave-one-out)
    correct_r1 = 0
    correct_r5 = 0
    for cls_idx in range(num_classes):
        mask = (all_labels == cls_idx)
        cls_embs = all_embeddings[mask]
        n = cls_embs.size(0)
        if n < 2:
            continue

        # For each sample in this class, compute leave-one-out centroid
        cls_sum = class_sums[cls_idx]
        # Centroids for other classes stay the same
        other_centroids = []
        other_indices = []
        for other_idx in range(num_classes):
            if other_idx == cls_idx:
                continue
            if class_counts[other_idx] < 1:
                continue
            other_centroids.append(class_sums[other_idx] / class_counts[other_idx])
            other_indices.append(other_idx)

        if not other_centroids:
            continue

        other_centroids = torch.stack(other_centroids, dim=0)  # (C-1, D)
        other_centroids_norm = F.normalize(other_centroids, dim=1)

        for i in range(n):
            # Leave-one-out centroid for this class
            loo_centroid = (cls_sum - cls_embs[i]) / (n - 1)
            # All centroids: this class (leave-one-out) + others
            all_cents = torch.cat([
                F.normalize(loo_centroid.unsqueeze(0), dim=1),
                other_centroids_norm,
            ], dim=0)

            emb_norm = F.normalize(cls_embs[i].unsqueeze(0), dim=1)
            sims = torch.mm(emb_norm, all_cents.t()).squeeze(0)
            # Rank-1: argmax must be index 0 (true-class LOO centroid)
            if sims.argmax().item() == 0:
                correct_r1 += 1
            # Rank-5: index 0 must be in the top-5 by similarity
            k = min(5, sims.size(0))
            _, topk_idx = sims.topk(k)
            if 0 in topk_idx.tolist():
                correct_r5 += 1

    accuracy = correct_r1 / max(1, total)
    accuracy_rank5 = correct_r5 / max(1, total)

    # Phase 3: Compile
    avg_latency = statistics.mean(latencies_ms)
    med_latency = statistics.median(latencies_ms)
    p95_idx = int(0.95 * (len(latencies_ms) - 1))
    p95_latency = sorted(latencies_ms)[p95_idx]
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0

    result = {
        "accuracy": round(accuracy, 4),
        "rank1_accuracy": round(accuracy, 4),
        "rank5_accuracy": round(accuracy_rank5, 4),
        "avg_latency_ms": round(avg_latency, 3),
        "median_latency_ms": round(med_latency, 3),
        "p95_latency_ms": round(p95_latency, 3),
        "fps": round(fps, 3),
        "model_size_mb": wrapped_model.size_mb(),
        "parameter_count": wrapped_model.param_count(),
        "num_samples": total,
        "num_classes": num_classes,
    }
    if constrained:
        result["peak_inference_ram_mb"] = round(peak_ram_mb, 1)
        result["peak_inference_delta_ram_mb"] = round(peak_delta_ram_mb, 1)
    return result


def _load_group_labels(csv_path):
    """Load subject -> {attribute: value} mapping from a CSV file."""
    if not os.path.exists(csv_path):
        return {}
    mapping = {}
    with open(csv_path, "r") as f:
        header = f.readline().strip().split(",")
        for line in f:
            parts = line.strip().split(",")
            if not parts or not parts[0]:
                continue
            row = dict(zip(header, parts))
            subj = row.pop(header[0])
            mapping[subj] = row
    return mapping


def benchmark_droneface_by_condition(wrapped_model, dataset, group_csv=None):
    """Break down DroneFace accuracy by height, distance, and group attributes."""
    if not hasattr(dataset, "metadata"):
        return {}

    group_map = _load_group_labels(group_csv) if group_csv else {}
    group_attrs = []
    if group_map:
        sample_row = next(iter(group_map.values()))
        group_attrs = list(sample_row.keys())

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    all_embeddings = []
    all_labels = []

    for images, labels in loader:
        emb = wrapped_model.get_embeddings(images)
        if not isinstance(emb, torch.Tensor):
            emb = torch.from_numpy(emb)
        all_embeddings.append(emb.cpu().float())
        all_labels.append(labels)

    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    num_classes = len(dataset.classes)
    class_sums = torch.zeros(num_classes, all_embeddings.size(1))
    class_counts = torch.zeros(num_classes)
    for i in range(all_embeddings.size(0)):
        lbl = all_labels[i].item()
        class_sums[lbl] += all_embeddings[i]
        class_counts[lbl] += 1

    height_results = defaultdict(lambda: {"correct": 0, "total": 0})
    distance_results = defaultdict(lambda: {"correct": 0, "total": 0})
    group_results = {attr: defaultdict(lambda: {"correct": 0, "total": 0})
                     for attr in group_attrs}

    for i in range(all_embeddings.size(0)):
        lbl = all_labels[i].item()
        meta = dataset.get_metadata(i)

        n_cls = class_counts[lbl].item()
        if n_cls < 2:
            continue

        loo_centroid = (class_sums[lbl] - all_embeddings[i]) / (n_cls - 1)

        centroids_list = []
        for c in range(num_classes):
            if c == lbl:
                centroids_list.append(loo_centroid)
            elif class_counts[c] >= 1:
                centroids_list.append(class_sums[c] / class_counts[c])
            else:
                centroids_list.append(torch.zeros(all_embeddings.size(1)))

        centroids = torch.stack(centroids_list, dim=0)
        emb_norm = F.normalize(all_embeddings[i].unsqueeze(0), dim=1)
        cent_norm = F.normalize(centroids, dim=1)
        sims = torch.mm(emb_norm, cent_norm.t()).squeeze(0)
        predicted = sims.argmax().item()
        is_correct = int(predicted == lbl)

        h = meta.get("height_m")
        if h is not None:
            key = "%.1fm" % h
            height_results[key]["correct"] += is_correct
            height_results[key]["total"] += 1

        d = meta.get("distance_m")
        if d is not None:
            bucket = "%.0fm" % (round(d / 2.0) * 2.0)
            distance_results[bucket]["correct"] += is_correct
            distance_results[bucket]["total"] += 1

        subj = meta.get("subject")
        if subj and subj in group_map:
            for attr in group_attrs:
                val = group_map[subj].get(attr)
                if val:
                    group_results[attr][val]["correct"] += is_correct
                    group_results[attr][val]["total"] += 1

    height_acc = {}
    for k, v in sorted(height_results.items()):
        if v["total"] > 0:
            height_acc[k] = round(v["correct"] / v["total"], 4)

    distance_acc = {}
    for k, v in sorted(distance_results.items(), key=lambda x: float(x[0][:-1])):
        if v["total"] > 0:
            distance_acc[k] = round(v["correct"] / v["total"], 4)

    group_acc = {}
    for attr, buckets in group_results.items():
        attr_acc = {}
        for k, v in sorted(buckets.items()):
            if v["total"] > 0:
                attr_acc[k] = {
                    "accuracy": round(v["correct"] / v["total"], 4),
                    "n": v["total"],
                }
        if attr_acc:
            group_acc[attr] = attr_acc

    out = {
        "accuracy_by_height": height_acc,
        "accuracy_by_distance": distance_acc,
    }
    if group_acc:
        out["accuracy_by_group"] = group_acc
    return out


# ── Main ─────────────────────────────────────────────────────────────────────

def print_results_table(results):
    print("\n" + "=" * 105)
    print("  BENCHMARK RESULTS")
    print("=" * 105)
    print("%-22s %8s %8s %10s %10s %8s %10s %12s" % (
        "Model", "Rank-1", "Rank-5", "Lat(ms)", "P95(ms)", "FPS", "Size(MB)", "Params"))
    print("-" * 105)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["accuracy"]):
        print("%-22s %7.2f%% %7.2f%% %10.1f %10.1f %8.1f %10.3f %12s" % (
            name,
            r["accuracy"] * 100,
            r.get("rank5_accuracy", r["accuracy"]) * 100,
            r["avg_latency_ms"],
            r["p95_latency_ms"],
            r["fps"],
            r["model_size_mb"],
            "{:,}".format(r["parameter_count"]),
        ))
    print("=" * 105)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", choices=list(MODEL_REGISTRY.keys()),
        help="Which recognition model to benchmark.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Benchmark ALL models in the registry.",
    )
    parser.add_argument(
        "--dataset-root", required=True,
        help="Path to dataset root folder.",
    )
    parser.add_argument(
        "--dataset-type", choices=["vggface2", "droneface"], required=True,
        help="Which dataset format to use.",
    )
    parser.add_argument(
        "--split", default="val",
        help="Dataset split for VGG Face2 (default: val). Ignored for DroneFace.",
    )
    parser.add_argument(
        "--constrained", action="store_true",
        help="Apply drone hardware constraints before benchmarking.",
    )
    parser.add_argument(
        "--cpu-mhz", type=int, default=None,
        help="Drone CPU target in MHz (default: 400). Use for CPU ablation.",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Device (default: cpu). Only affects PyTorch models.",
    )
    parser.add_argument(
        "--output-dir", default="benchmark_results",
        help="Directory to write JSON results.",
    )
    parser.add_argument(
        "--group-csv", default="droneface_groups.csv",
        help="CSV mapping DroneFace subject -> group attributes (gender, etc.).",
    )
    parser.add_argument(
        "--save-embeddings-dir", default=None,
        help="If set, saves per-(model,dataset) embeddings/labels/filenames here.",
    )
    args = parser.parse_args()

    if not args.model and not args.all:
        parser.error("Specify --model or --all")

    if args.constrained:
        from drone_constraints import print_constraint_summary, set_drone_cpu_mhz
        if args.cpu_mhz is not None:
            set_drone_cpu_mhz(args.cpu_mhz)
        print_constraint_summary()

    model_names = list(MODEL_REGISTRY.keys()) if args.all else [args.model]

    all_results = {}
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for model_name in model_names:
        factory, desc = MODEL_REGISTRY[model_name]
        print("\n[bench] === %s (%s) ===" % (model_name, desc))

        try:
            wrapped_model = factory()
        except Exception as e:
            import traceback
            print("  [bench] ERROR loading %s: %s" % (model_name, e))
            traceback.print_exc()
            print("  [bench] Skipping this model.")
            continue

        # Load dataset with the model's expected input size
        input_size = wrapped_model.input_size
        print("[bench] Loading dataset: %s (input size: %dx%d)" % (
            args.dataset_type, input_size, input_size))

        if args.dataset_type == "vggface2":
            dataset = VGGFace2Dataset(
                args.dataset_root, split=args.split, input_size=input_size)
        else:
            dataset = DroneFaceDataset(args.dataset_root, input_size=input_size)

        print("[bench] %d images, %d identities" % (len(dataset), len(dataset.classes)))

        result = benchmark_model(
            wrapped_model, dataset,
            constrained=args.constrained,
            save_embeddings_dir=args.save_embeddings_dir,
            dataset_name=args.dataset_type,
        )
        result["model_name"] = model_name
        result["description"] = desc
        result["dataset"] = args.dataset_type
        result["constrained"] = args.constrained
        result["input_size"] = input_size
        if args.constrained:
            from drone_constraints import DRONE_CPU_MHZ, DRONE_RAM_MB
            result["cpu_mhz_target"] = DRONE_CPU_MHZ
            result["ram_mb_limit"] = DRONE_RAM_MB

        if args.dataset_type == "droneface":
            print("  [bench] Computing per-condition breakdown...")
            conditions = benchmark_droneface_by_condition(
                wrapped_model, dataset, group_csv=args.group_csv)
            result["conditions"] = conditions

            if conditions.get("accuracy_by_height"):
                print("  [bench] Accuracy by height:")
                for h, acc in conditions["accuracy_by_height"].items():
                    print("    %s : %.2f%%" % (h, acc * 100))
            if conditions.get("accuracy_by_distance"):
                print("  [bench] Accuracy by distance:")
                for d, acc in conditions["accuracy_by_distance"].items():
                    print("    %s : %.2f%%" % (d, acc * 100))
            if conditions.get("accuracy_by_group"):
                for attr, buckets in conditions["accuracy_by_group"].items():
                    print("  [bench] Accuracy by %s:" % attr)
                    for val, stats in buckets.items():
                        print("    %s (n=%d) : %.2f%%" % (
                            val, stats["n"], stats["accuracy"] * 100))

        all_results[model_name] = result

        if args.constrained:
            from drone_constraints import DRONE_CPU_MHZ
            suffix = "constrained_%dmhz" % DRONE_CPU_MHZ
        else:
            suffix = "unconstrained"
        fname = "bench_%s_%s_%s.json" % (model_name, args.dataset_type, suffix)
        with open(str(out_dir / fname), "w") as f:
            json.dump(result, f, indent=2)
        print("  [bench] Saved: %s" % (out_dir / fname))

    if all_results:
        print_results_table(all_results)

        if args.constrained:
            from drone_constraints import DRONE_CPU_MHZ
            suffix = "constrained_%dmhz" % DRONE_CPU_MHZ
        else:
            suffix = "unconstrained"
        combined_path = out_dir / ("benchmark_combined_%s_%s.json" % (
            args.dataset_type, suffix))
        with open(str(combined_path), "w") as f:
            json.dump(all_results, f, indent=2)
        print("[bench] Combined results: %s" % combined_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
