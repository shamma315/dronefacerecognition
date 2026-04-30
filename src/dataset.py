import os
import shutil
import random
from PIL import Image
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
                if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(class_Folder, img))
                    self.labels.append(label)
                
        #applying data transformations 

        train_transform = transforms.Compose([transforms.RandomResizedCrop(112, scale=(0.8,1.0)),
                                              transforms.RandomHorizontalFlip(),
                                             transforms.RandomRotation(20),
                                             transforms.GaussianBlur(kernel_size=3),
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
resnet_model = InceptionResnetV1(pretrained='vggface2')


def get_embedding(img_path, model_name="DeepFace", detector="retinaface"):
     
    try: 
        img = Image.open(img_path).convert('RGB')
        transform = transforms.Compose([transforms.Resize((160,160)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5,0.5])])
        
        img_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            embed = resnet_model(img_tensor)
            embed = embed/np.linalg.norm(embed) #l2 normalized embedding
        return embed.squeeze().numpy()
    except Exception as e: 
         print(f" Skipped {os.path.basename(img_path)}: {e}")
         return None
    
def identity_SPLIT(root_dir, train_ratio=0.8, val_ratio=0.1):
    all_ids= sorted([n for n in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, n))])
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

class VGGFaceDataset(Dataset):
     def __init__(self, root_dir, selected_ids,id_map, augment=True):
          self.root_dir= root_dir
          self.selected_ids = selected_ids
          self.id_map = id_map
          
          self.image_paths = []
          self.labels = []
          for id_name in selected_ids:
               class_folder = os.path.join(root_dir, id_name)
               if not os.path.isdir(class_folder):
                    continue
               for img in os.listdir(class_folder):
                    if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                         self.image_paths.append(os.path.join(class_folder, img))
                         self.labels.append(self.id_map[id_name])
                    
                    
          
          self.train_transform = transforms.Compose([transforms.Resize((128,128)),
                                               transforms.RandomResizedCrop(112, scale=(0.6, 1.0)), #simulating drone footage 
                                               transforms.RandomHorizontalFlip(),
                                               transforms.RandomRotation(25), #overhead angle
                                               transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2),
                                                transforms.GaussianBlur(kernel_size = 3, sigma=(0.1, 2.0)), #motion blur
                                                transforms.RandomGrayscale(p=0.1),
                                                transforms.ToTensor(),
                                               transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                               transforms.RandomErasing(p=0.2,scale=(0.02, 0.15)), #occlusion
                                              ])
          self.val_transform = transforms.Compose([transforms.Resize((112, 112)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),])
          
          self.augment = augment

     def __len__(self):
          return len(self.image_paths)
     def __getitem__(self, idx):
          img = Image.open(self.image_paths[idx]).convert("RGB")
          label = self.labels[idx]
          if self.augment:
               img=self.train_transform(img)
          else:
               img = self.val_transform(img) 
          return img, label
     

     


