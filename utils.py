import torch
from torchvision import transforms, datasets
import os

data_transform = transforms.Compose(
    [transforms.Resize((256,256)),
    transforms.ToTensor()]
)

dataset = datasets.ImageFolder(root = 'real_images',
transform = data_transform)

dataset_path = 'real_images'
pallets = len(os.listdir(os.path.join(dataset_path, 'pallet')))
forklifts = len(os.listdir(os.path.join(dataset_path, 'forklift')))
person = len(os.listdir(os.path.join(dataset_path, 'person')))

tot = pallets + forklifts + person
train_ratio = 0.8
num_train_samples = int(0.8*tot) 

training_dataset, validation_dataset = torch.utils.data.random_split(dataset, lengths = [num_train_samples, tot - num_train_samples])
