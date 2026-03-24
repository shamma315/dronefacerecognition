import os
import shutil
import random
from PIL import Image
from deepface import DeepFace 
import torch
import numpy as np
from torch.utils.data import Dataset 
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1


class DroneFaceDataset(Dataset):
    def __init__(self, root_dir, augment=True): 
        self.root_dir = root_dir
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.image_paths = []
        self.labels = []

        for label, class_name in  enumerate(self.classes):
            class_Folder = os.path.join(root_dir, class_name)
            for img in os.listdir(class_Folder):
                self.image_paths.append(os.path.join(class_Folder, img))
                self.labels.append(label)
                
        #applying data transformations 

        train_transform = transforms.Compose([transforms.Resize((112, 112)), 
                                             transforms.RandomHorizontalFlip(),
                                             transforms.RandomRotation(20),
                                             transforms.GaussianBlur(kernel_size=3),
                                             transforms.RandomResizedCrop(112, scale=(0.8,1.0)),
                                             transforms.ColorJitter(brightness=0.2, contrast=0.2),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                  std = [0.5, 0.5, 0.5]),
                                            ])
        val_transform = transforms.Compose([transforms.Resize((112, 112)), 
                                            transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                                  std = [0.5, 0.5, 0.5]),])   #to test real performance
        self.transform = train_transform if augment else val_transform
     
                                            
    def __len__(self):
            return len(self.image_paths)
        
    def __getitem__(self, idx):
            img = Image.open(self.image_paths[idx]).convert("RGB")
            label = self.labels[idx]
            img = self.transform(img)

            return img, label


#loading pretrained model 
resnet_model = InceptionResnetV1(pretrained='vggface2').eval()   
def get_embedding(img_path, model_name="DeepFace", detector="retinaface"):
     
    try: 
        img = Image.open(img_path).convert('RGB')
        img = img.resize((160, 160)) # size required for the model
        img_tensor = torch.tensor(np.array(img)).permute(2,0,1).unsqueeze(0).float() / 255.0
        with torch.no_grad():
            embed = resnet_model(img_tensor)
        
        #  result = DeepFace.represent(
        #       img_path = img_path, 
        #       model_name = "ArcFace",
        #       detector_backend = detector, 
        #       enforce_detection = False,
        #       align =True,
        #  )
        #  embed = np.array(result[0]["embedding"], dtype=np.float32)
        embed = embed/np.linalg.norm(embed) #l2 normalized embedding
        return embed.squeeze().numpy()
    except Exception as e: 
         print(f" Skipped {os.path.basename(img_path)}: {e}")
         return None
    
def UAV_SPLIT(root_dir, train_ratio=0.8):
     #splitting dataset 2:UAV_Human dataset folders by identity 
     #identities: folders with P
     all_ids = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
     np.random.seed(42)
