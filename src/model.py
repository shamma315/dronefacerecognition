import torch
import torch.nn as nn 
import torch.nn.functional as F

class Embeddinghead(nn.Module):
    def __init__(self, input_dim=512, Embedding_dim=256):
        super().__init__()

       
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x=self.fc1(x)
        x=self.bn1(x)
        x=F.relu(x) #activation function to keep only positive values
        x=self.dropout(x)
        x=self.fc2(x)
        x=self.bn2(x)
        x=F.normalize(x, p=2, dim=1) #keeping embeddings same length
        return x 

class ArcFaceLoss(nn.Module):

    def __init__(self, in_features, num_classes, scale=64.0, margin=0.5): 
        # in_features = embedding size
        # num_classes=number of classes in our dataset
        # s = scaling factor
        # m = angular margin
        super().__init__()
        self.s = scale
        self.m = margin
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight) #one learnable vector per person in embedding space
    def forward(self, embeddings, labels):
        cosine_theta = F.linear(F.normalize(embeddings), F.normalize(self.weight)) #measures similarity
        theta = torch.acos(cosine_theta.clamp(-1 + 1e-7,  1 - 1e-7)) #converting to an angle
        one_hot_encode = torch.zeros_like(cosine_theta) #mask creation for image classification
        one_hot_encode.scatter_(1, labels.view(-1,1), 1)
        output = torch.where(one_hot_encode.bool(), torch.cos(theta 
                                                            + self.m), cosine_theta)  #improves model training
        
        return output * self.s #scaled by 64 for effective training
    

        
