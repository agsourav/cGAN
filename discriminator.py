import torch
import torch.nn as nn

def conv2d(Cin, Cout, k_size, pad, stride):
    layer = nn.Sequential(
        nn.Conv2d(in_channels = Cin, out_channels= Cout, kernel_size = k_size, padding= pad,
        stride = stride),
        nn.LeakyReLU(inplace = True)
    )
    return layer

def conv_block(Cin, Cout, k_size, pad, stride):
    layers = nn.Sequential(
        nn.Conv2d(Cin, Cout, kernel_size = k_size, padding = pad, stride = stride),
        nn.BatchNorm2d(Cout),
        nn.LeakyReLU(inplace = True)
    )
    return layers

def final(Cin, Cout, num_features):
    layers = nn.Sequential(
        nn.Conv2d(in_channels = Cin, out_channels= Cout, kernel_size = 3, padding= 1, stride = 2),
        nn.Flatten(),
        nn.Linear(num_features, 256),
        nn.ReLU(inplace = True),
        nn.Linear(256, 32),
        nn.Linear(32, 1),
        nn.Sigmoid()
    )

    return layers

class CGANDiscriminator(nn.Module):
    def __init__(self, num_classes, dis_inp):
        super(CGANDiscriminator, self).__init__()
        self.dis_inp = dis_inp
        self.label_embedding = nn.Linear(num_classes, dis_inp*dis_inp)
        self.image_dropout = nn.Dropout(p = 0.25)
        self.conv2d_1 = conv2d(4, 64, k_size = 3, pad = 1, stride = 1)
        self.conv_block1 = conv_block(64, 128, k_size = 1, pad = 0, stride = 2)     #downsampling
        self.conv_block2 = conv_block(128, 256, k_size = 1, pad = 0, stride = 1)    
        self.conv2d_2 = conv2d(256, 256, k_size = 1, pad = 0, stride = 2)           #downsampling
        self.conv2d_3 = conv2d(256, 256, k_size = 1, pad = 0, stride = 2)           #downsampling
        self.conv_block3 = conv_block(256, 256, k_size = 3, pad = 0, stride = 1)
        self.final = final(256, 512, num_features = 512*15*15)

    def forward(self, x, labels):

        embedded_labels = self.label_embedding(labels)
        inp_img = self.image_dropout(x)

        N = x.shape[0]
        h = self.dis_inp
        w = h
        l = torch.reshape(embedded_labels, (N, 1, h, w))
        inp = torch.cat([inp_img, l], dim = 1)

        layers = [self.conv2d_1, self.conv_block1, self.conv_block2, self.conv2d_2, self.conv2d_3,
        self.conv_block3, self.final]
        model = nn.Sequential(*layers) 
        
        disOut = -1
        try:
            disOut = model(inp)
        except Exception as e:
            print("Incorrect dim!!")
            print("----------------------------------")
            print(e)
        return disOut


