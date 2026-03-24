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
import pandas as pd

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
    
def bird_SPLIT(root_dir, train_ratio=0.8, val_ratio=0.1):
    df = pd.read_csv(root_dir)
    #each source acting as a unique id
    all_ids = df['source_id'].unique().tolist()
    np.random.seed(42)
    np.random.shuffle(all_ids)
    num_train = int(len(all_ids) * train_ratio)
    num_val = int(len(all_ids) * val_ratio)
    train_ids = all_ids[:num_train]
    val_ids = all_ids[num_train : num_train + num_val]
    test_ids = all_ids[num_train + num_val:]
    print(f"training_ids: {len(train_ids)}")
    print(f"val_ids: {len(val_ids)}")
    print(f"test_ids: {len(test_ids)}")
    return train_ids, val_ids, test_ids

class birdsEyeDataset(Dataset):
     def __init__(self, root_dir, label_dir, selected_ids, df_metadata, augment=True):
          self.root_dir= root_dir
          self.label_dir = label_dir
          
          #filtering metadata for selected identities
          self.relevant_metadata = df_metadata[df_metadata['source_id'].isin(selected_ids)]
          self.image_Files = self.relevant_metadata['filename'].tolist()

          self.id_map = {id_name: i for i, id_name in enumerate(selected_ids)}

          self.transform = transforms.Compose([transforms.Resize((112,112)),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.5], [0.5])])

        #   for idx, class_name in enumerate(self.classes):
        #        class_folder = os.path.join(root_dir, class_name)
        #        images = []
        #        for root, _, files in os.walk(class_folder):
        #             for f in files: 
        #                  if f.lower().endswith(('.png', '.jpg', '.jpeg')):
        #                       images.append(os.path.join(root, f))

        #        if len(images) > max_images_per_id:
        #             images= np.random.choice(images, max_images_per_id, replace=False)
        #        for img_path in images: 
        #             self.image_paths.append(img_path)
        #             self.labels.append(idx)
        #   train_transform = transforms.Compose([transforms.Resize((112, 112)), 
        #                                      transforms.RandomHorizontalFlip(),
        #                                      transforms.ColorJitter(brightness=0.1, contrast=0.1),
        #                                      transforms.ToTensor(),
        #                                      transforms.Normalize(mean=[0.5, 0.5, 0.5],
        #                                                           std = [0.5, 0.5, 0.5]),
        #                                     ])
        #   val_transform = transforms.Compose([transforms.Resize((112, 112)), 
        #                                     transforms.ToTensor(),
        #                                      transforms.Normalize(mean=[0.5, 0.5, 0.5],
        #                                                           std = [0.5, 0.5, 0.5]),])   #to test real performance
        #   self.transform = train_transform if augment else val_transform
     def __len__(self):
          return len(self.image_paths)
     def __getitem__(self, idx):
          img_name = self.image_Files[idx]
          img_path = os.path.join(self.img_dir, img_name)
          label_path = os.path.join(self.label_dir, img_name.replace('.jpg', '.txt'))
          full_img = Image.open(img_path).convert('RGB')
          w,h = full_img.size
          #For YOLO Image transformations 
          with open(label_path, 'r') as f: 
               line= f.readline().split()
               if not line: return self.__getitem__((idx+1) % len(self))
               _, x_c, y_c, bw, bh = map(float, line)

        # Convert from YOLO NORMALIZED TO PIXEL COORDINATES
          left = (x_c - bw/2) * w
          top = (y_c - bh/2) * h
          right = (x_c + bw/2) * w
          bottom = (y_c + bh/2) * h

          face_Crop = full_img.crop((left, top, right, bottom))
          face_tensor = self.transform(face_Crop)

          #get target ID label 
          source_id = self.relevant_meta.iloc[idx]['source_id']
          label = self.id_map[source_id]
          return face_tensor, label 


     


