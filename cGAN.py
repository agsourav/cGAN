import torch
import torch.nn as nn
import torch.optim as optim
from generator import cGANGenerator
from discriminator import CGANDiscriminator
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os

try:
    matplotlib.use('TkAgg')
except Exception as e:
    print(e)

eps = 1e-10
class cGAN():
    def __init__(self, num_classes, gen_inp, dis_inp, latent_size):

        self.gen = cGANGenerator(num_classes, latent_size, gen_inp)
        self.dis = CGANDiscriminator(num_classes, dis_inp)
        self.dis = nn.DataParallel(self.dis)
        self.gen = nn.DataParallel(self.gen)
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
            self.genLosses = self.gen_check['loss']
            self.genScores = self.gen_check['score']

    def get_genScore(self, z, c):
        gen_out = self.gen(z,c)
        genScore = self.dis(gen_out, c)
        return genScore

    def get_disScore(self, x, labels):
        disScore = self.dis(x, labels)
        return disScore

    def train(self, train_loader, epochs, num_iters, gen_lr, dis_lr, dis_epochs, num_classes, loss_func, device):
        
        self.dis.to(device)
        self.gen.to(device)
        
        optim_G = optim.Adam(self.gen.parameters(), lr = gen_lr, betas = (0.5, 0.999))
        optim_D = optim.Adam(self.dis.parameters(), lr = dis_lr, betas = (0.5, 0.999))
        
        
        start_epoch = 0
        if self.gen_check != '':
            start_epoch = self.gen_check['epoch']
        end_epoch = start_epoch + epochs
        for i in range(start_epoch, end_epoch):
            dataiter = iter(train_loader)
            print('Epoch: {0}'.format(i+1))
            disloss_eps = np.array([])
            genloss_eps = np.array([])
            disscore_eps = np.array([])
            genscore_eps = np.array([])
            for k in range(num_iters):
                #print('\titeration: {0}'.format(k+1))
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
                        genScore = self.get_genScore(z, labels)
                    except Exception as e:
                        print(e)
                        exit(0)
                    try:
                        if loss_func == 'l2':
                            disloss = (disScore.mean() - 1.0).pow(2) + (genScore.mean()).pow(2)
                        else:
                            disloss = -torch.log(disScore.mean() + eps) - torch.log(1.0 - genScore.mean() + eps) 
                        #print('dis iter: {0}\tdiscriminator loss {1}'.format(j, disloss))
                        disloss.backward()
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
                
                disscore_eps = np.append(disscore_eps, disScore.mean().item())
                genscore_eps = np.append(genscore_eps, genScore.mean().item())

                if loss_func == 'l2':
                    disloss = (disScore.mean() - 1.0).pow(2) + (genScore.mean()).pow(2)
                    genloss = (genScore.mean() - 1.0).pow(2)
                else:
                    disloss = -torch.log(disScore.mean() + eps) - torch.log(1.0 - genScore.mean() + eps)
                    genloss = -torch.log(genScore.mean() + eps)

                disloss_eps = np.append(disloss_eps, disloss.item())
                genloss_eps = np.append(genloss_eps, genloss.item())
                #disloss = -disScore - (1.0 - genScore) 
                #genloss = -genScore
                if k%5==0:
                    print('  iteration: {0}'.format(k))
                    print('\tdiscriminator score: {0:.5f}\n\tgenerator score: {1:.5f}'.format(disScore.mean().item(), genScore.mean().item()))
                    print('\tdiscriminator loss {0:.5f}\n\tgenerator loss: {1:.5f}'.format(disloss.item(), genloss.item()))

                disloss.backward(retain_graph = True)
                genloss.backward()
                optim_D.step()
                optim_G.step()
                
            self.disLosses = np.append(self.disLosses, disloss_eps.mean())
            self.genLosses = np.append(self.genLosses, genloss_eps.mean())
            self.genScores = np.append(self.genScores, genscore_eps.mean())
            self.disScores = np.append(self.disScores, disscore_eps.mean())

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

    def display_images(self, inp, labels, file_name):
        for i in range(inp.shape[0]):
            plt.imshow(inp[i].permute(1,2,0).detach().numpy())
            plt.title(str(labels[i]))
            file_path = 'output/' + file_name + str(i) + '_1.png'
            plt.savefig(file_path)

    def plot_(self, s = 'scores'):
        fig = plt.figure()
        if s == 'losses':
            plt.plot(self.disLosses, 'r');
            plt.plot(self.genLosses, 'g');
            plt.legend(['discriminator losses', 'generator losses']);
            plt.title('losses vs epochs');
            plt.savefig('output/losses.png')
            plt.show()
        else:
            plt.plot(self.disScores, 'r');
            plt.plot(self.genScores, 'g');
            plt.legend(['discriminator scores', 'generator scores']);
            plt.title('scores vs epochs');
            plt.savefig('output/scores.png')
            plt.show()

