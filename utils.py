import torch
from torchvision import transforms, datasets
import os
import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description= 'Conditional GAN')
    parser.add_argument('--image-dir', dest = 'image_dir', help = 'specify the path to images',
    default = 'real_images', type = str)
    parser.add_argument('--eps', dest = 'epochs', help = 'number of epochs to train')
    parser.add_argument('--gen-lr', dest = 'gen_lr', help = 'generator learning rate',
    default = 0.0002)
    parser.add_argument('--dis-lr', dest = 'dis_lr', help = 'discriminator learning rate',
    default = 0.0001)
    parser.add_argument('--train', dest = 'train', help = '1 to train/0 to infer', default = 1)
    parser.add_argument('--bs', dest = 'batch_size', help = 'batch size for training', default = 4)
    parser.add_argument('--dis-eps', dest = 'dis_eps', help = 'number of iter for discriminator for each iter of generator',
    default = 2)
    return parser.parse_args()

