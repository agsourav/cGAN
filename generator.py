import torch
import torch.nn as nn
import numpy as np

def transpose2d(in_ch, out_ch, k_sizes = [5,3], dil = [2,1], ops = [0,1]):
    layer = nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size = k_sizes[0], stride = 1, padding = 0,
        output_padding = ops[0], dilation = dil[0]),
        nn.BatchNorm2d(out_ch),
        nn.ConvTranspose2d(out_ch, out_ch, kernel_size = k_sizes[1], stride = 2, padding = 0,
        output_padding = ops[1], dilation = dil[1]),
        nn.ReLU(inplace = True)
    )
    return layer

def transposeF(in_ch, out_ch = 3, kernel_size = 3, dil = 1, op = 0, pad = 0, stride = 1):
    layer = nn.Sequential(
        nn.ConvTranspose2d(in_channels = in_ch, out_channels= out_ch, kernel_size = kernel_size,
        padding = pad, stride = stride, output_padding= op, dilation = dil),
        nn.BatchNorm2d(out_ch),
        nn.Tanh()
    )
    return layer


class cGANGenerator(nn.Module):
    def __init__(self, num_classes, latent_size, gen_inp):
        super(cGANGenerator, self).__init__()
        self.gen_inp = gen_inp
        self.latent_embedding = nn.Sequential(
            nn.Linear(latent_size, gen_inp)
        ) 

        self.condition_embedding = nn.Sequential(
            nn.Linear(num_classes, gen_inp)
        )

    def forward(self, x, con):
        z = self.latent_embedding(x)
        c = self.condition_embedding(con)
        h = int(np.sqrt(self.gen_inp))
        w = h
        N = z.shape[0]      #batch_size
        z = torch.reshape(z, (N,1,h,w))
        c = torch.reshape(c, (N,1,h,w))
        
        inp = torch.cat([z,c], dim = 1)     #(N, C, Hin, Win) 
        transpose_layers = [transpose2d(inp.shape[1], inp.shape[1]), transpose2d(inp.shape[1], inp.shape[1]),
        transpose2d(inp.shape[1], inp.shape[1])]        #(Nx2x16x16) -> (Nx2x254x254)
        transF = [transposeF(inp.shape[1])]             #(Nx2x254x254) -> (Nx3x256x256)
        generator_layers = transpose_layers + transF
        model = nn.Sequential(*generator_layers)
        if inp.is_cuda:
            model.cuda()
        gen_image = model(inp)              
        return gen_image
    

