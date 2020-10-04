import torch
import torch.nn as nn
import torch.optim as optim
from generator import cGANGenerator
from discriminator import CGANDiscriminator
import numpy as np
import matplotlib.pyplot as plt
import os

class cGAN():
    def __init__(self, num_classes, gen_inp, dis_inp, latent_size):

        self.gen = cGANGenerator(num_classes, latent_size, gen_inp)
        self.dis = CGANDiscriminator(num_classes, dis_inp)
        self.dis_path = 'weights/discriminator.pt'
        self.gen_path = 'weights/generator.pt'
        self.dis_check = ''
        self.gen_check = ''
        self.genScores, self.genLosses, self.disScores, self.disLosses = np.array([]), np.array([]), np.array([]), np.array([])

        if os.path.exists(self.dis_path):
            self.dis_check = torch.load(self.dis_path)
            self.dis.load_state_dict(self.dis_check['model_state_dict'])
            self.disLosses = self.dis_check['loss']
            self.disScores = self.dis_check['score']

        if os.path.exists(self.gen_path):
            self.gen_check = torch.load(self.gen_path)
            self.gen.load_state_dict(self.gen_check['model_state_dict'])
            self.disLosses = self.gen_check['loss']
            self.disScores = self.gen_check['score']

    def get_genScore(self, z, c):
        gen_out = self.gen(z,c)
        print(gen_out.is_cuda)
        genScore = self.dis(gen_out, c)
        return genScore

    def get_disScore(self, x, labels):
        disScore = self.dis(x, labels)
        print(disScore.is_cuda)
        return disScore

    def train(self, train_loader, epochs, num_iters, gen_lr, dis_lr, dis_epochs, num_classes, device):

        self.dis.to(device)
        self.gen.to(device)
        
        optim_G = optim.Adam(self.gen.parameters(), lr = gen_lr)
        optim_D = optim.Adam(self.dis.parameters(), lr = dis_lr)
        
        dataiter = iter(train_loader)
        start_epoch = 0
        if self.gen_check != '':
            start_epoch = self.gen_check['epoch']
        end_epoch = start_epoch + epochs
        for i in range(start_epoch, end_epoch):
            print('Epoch: {0}'.format(i+1))
            for k in range(num_iters):
                print('\titeration: {0}'.format(k+1))
                for j in range(dis_epochs-1):
                    optim_D.zero_grad()
                    real_images, labels = next(dataiter)
                    labels = nn.functional.one_hot(labels, num_classes)
                    labels = labels.float()
                    batch_size = real_images.shape[0]
                    z = torch.randn(size = (batch_size, 1))
                    real_images, labels, z = real_images.to(device), labels.to(device), z.to(device)
                    #print(next(self.dis.parameters()).is_cuda, next(self.gen.parameters()).is_cuda)
                    try:
                        disScore = self.get_disScore(real_images, labels)
                        print(disScore)
                        genScore = self.get_genScore(z, labels)
                    except Exception as e:
                        print(e)
                        exit(0)
                    try:
                        disloss = -disScore - (1 - genScore) 
                        #print('dis iter: {0}\tdiscriminator loss {1}'.format(j, disloss))
                        disloss.mean().backward()
                        optim_D.step()
                    except Exception as e:
                        print(e)
                        break
                optim_D.zero_grad()
                optim_G.zero_grad()
                real_images, labels = next(dataiter)
                batch_size = real_images.shape[0]
                labels = nn.functional.one_hot(labels, num_classes)
                labels = labels.float()
                z = torch.randn(size = (batch_size, 1))
                real_images, labels, z = real_images.to(device), labels.to(device), z.to(device)
                disScore = self.get_disScore(real_images, labels)
                genScore = self.get_genScore(z, labels)
                
                disloss = -disScore - (1.0 - genScore) 
                genloss = -genScore
                if k%5==0:
                    print('\tdiscriminator score: {0}\n\tgenerator score: {1}'.format(disScore, genScore))
                    print('discriminator loss {0}\tgenerator loss: {1}'.format(disloss.mean().item(), genloss.mean().item()))
                
                disloss.mean().backward(retain_graph = True)
                optim_D.step()
                genloss.mean().backward()
                optim_G.step()
            np.append(self.disLosses, disloss.mean().item())
            np.append(self.genLosses, genloss.mean().item())
            np.append(self.genScores, genScore.mean().item())
            np.append(self.disScores, disScore.mean().item())

            print('-'*20)
            if i%10==0:
                self.checkpoint(self.gen_path, i+1, self.gen, self.genLosses, self.genScores)
                self.checkpoint(self.dis_path, i+1, self.dis, self.disLosses, self.disScores)
                
        self.checkpoint(self.gen_path, i+1, self.gen, self.genLosses, self.genScores)
        self.checkpoint(self.dis_path, i+1, self.dis, self.disLosses, self.disScores)

    def checkpoint(self, path, epoch, model, loss, score):    
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': loss,
            'score': score
        }, path)
        print('checkpointing successful at {0} epoch'.format(epoch))

    def infer(self, labels):
        N = labels.shape[0]
        z = torch.randn(size = (N, 1))
        gen_images = self.gen(z, labels)
        return gen_images

    def display_images(self, inp):
        for i in range(inp.shape[0]):
            plt.imshow(inp[i].permute(1,2,0).detach().numpy())
            plt.show()

    def plot_(self, s = 'scores'):
        fig = plt.figure()
        if s == 'losses':
            plt.plot(self.disLosses, 'r');
            plt.plot(self.genLosses, 'g');
            plt.legend(['discriminator losses', 'generator losses']);
            plt.title('losses vs epochs');
            plt.show() 
        else:
            plt.plot(self.disScores, 'r');
            plt.plot(self.genScores, 'g');
            plt.legend(['discriminator scores', 'generator scores']);
            plt.title('scores vs epochs');
            plt.show()

