import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torchsummary import summary
from PIL import ImageFile
from cGAN import cGAN
from utils import *
import os

CUDA = torch.cuda.is_available()
device = torch.device('cuda' if CUDA else 'cpu')

ImageFile.LOAD_TRUNCATED_IMAGES = True
print(torch.__version__)

#arguments
args = arg_parse()
root_dir = args.image_dir
epochs = int(args.epochs)
gen_lr = float(args.gen_lr)
dis_lr = float(args.dis_lr)
train = int(args.train)
batch_size = int(args.batch_size)
dis_iter = int(args.dis_eps)

discriminator_inp = 256
generator_inp = 16*16
latent_size = 1

num_classes = len(os.listdir(root_dir))
pallets = len(os.listdir(os.path.join(root_dir, 'pallet')))
forklifts = len(os.listdir(os.path.join(root_dir, 'forklift')))
person = len(os.listdir(os.path.join(root_dir, 'person')))

tot = pallets + forklifts + person
train_ratio = 0.8
num_train_samples = int(0.8*tot) 

num_iters = num_train_samples//(batch_size*dis_iter)
print(num_iters)
data_transform = transforms.Compose(
    [transforms.Resize((256,256)),
    transforms.ToTensor()]
)

dataset = datasets.ImageFolder(root = root_dir,
transform= data_transform)

training_dataset, validation_dataset = torch.utils.data.random_split(dataset,
lengths = [num_train_samples, tot - num_train_samples])


if train:
    print(num_train_samples)
    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size = batch_size,
    shuffle = True, num_workers= 4)

    gan = cGAN(num_classes, generator_inp, discriminator_inp, latent_size)
    gan = nn.DataParallel(gan)
    gan.train(train_loader, epochs, num_iters, gen_lr, dis_lr, dis_iter, num_classes, device)
    gan.plot_()
    gan.plot_(s = 'losses')

else:
    file_name = args.file_name
    gan = cGAN(num_classes, generator_inp, discriminator_inp, latent_size)
    
    labels = torch.tensor([[0],[1],[2]])
    labels = nn.functional.one_hot(labels, num_classes)
    labels = labels.float()
    gen_images = gan.infer(labels = labels)
    gan.display_images(gen_images, labels, file_name)


