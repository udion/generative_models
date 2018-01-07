import os
import random
import torch as tch
import torch.nn as tchnn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as tchoptim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.autograd as tchagd
import torch.nn.functional as F

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
import scipy as sp
from scipy.misc import imresize

import visdom

z_indim = 100
ngf = 64
nc = 1
ndf = 64

#visdom client
vis = visdom.Visdom(port=7777)

#make dir for models if not already present
try:
	os.makedirs('./models')
except OSError as exception:
	if exception.errno != errno.EEXIST:
		raise
try:
	os.makedirs('./models/unimodal')
except OSError as exception:
	if exception.errno != errno.EEXIST:
		raise

class Generator(tchnn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = tchnn.Sequential(
            # input is Z, going into a convolution
            tchnn.ConvTranspose2d(z_indim, ngf*8, 4, 1, 0, bias=False),
            tchnn.BatchNorm2d(ngf*8),
            tchnn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            tchnn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            tchnn.BatchNorm2d(ngf*4),
            tchnn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            tchnn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            tchnn.BatchNorm2d(ngf*2),
            tchnn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            tchnn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            tchnn.BatchNorm2d(ngf),
            tchnn.ReLU(True),
            # state size. (ngf) x 32 x 32
            tchnn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            tchnn.Tanh()
            # state size. (nc) x 64 x 64
        )
    
    def forward(self, x):
        return self.net(x)
    
    def name(self):
        return 'GENERATOR'

class Discriminator(tchnn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = tchnn.Sequential(
            # input is (nc) x 64 x 64
            tchnn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            tchnn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            tchnn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            tchnn.BatchNorm2d(ndf*2),
            tchnn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            tchnn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            tchnn.BatchNorm2d(ndf*4),
            tchnn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            tchnn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            tchnn.BatchNorm2d(ndf*8),
            tchnn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            tchnn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            tchnn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)
    
    def name(self):
        return 'DISCRIMINATOR'

# custom weights initialization, taken from pytorch manual examples
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def mnist_resize(x_vec):
    rx = imresize(x_vec.reshape(-1,28), (ndf,ndf), interp='bicubic')
    rx = rx.astype(dtype='float')
    rx = rx/255.0
    rx_vec = rx.reshape(-1,ndf*ndf)
    return rx_vec

MNISTX_train = np.load('../data/MNIST/3/3.npy')
# MNISTy_train = np.load('../../gmmGAN/MNISTy_train.npy')
#resizing
rsz_MNISTX_train = np.zeros((MNISTX_train.shape[0], ndf*ndf))
for i in range(MNISTX_train.shape[0]):
    rsz_MNISTX_train[i,:] = mnist_resize(MNISTX_train[i,:])
print('data loaded...')
rsz_MNISTX_train.shape

def MNIST_gen(X, BATCHSZ):
    X = X.reshape(-1, ndf*ndf)
    while(True):
        databatch = random.sample(list(X), BATCHSZ)
        databatch = np.array(databatch)
        yield databatch

BATCHSZ = 128
MNISTd = MNIST_gen(rsz_MNISTX_train, BATCHSZ)

#mnist batch plotter
def plotter64(batch_data):
    #batch_data = batch_data.numpy()
    n = batch_data.shape[0]
    for i in range(n):
        plt.subplot(8,8,i+1)
        plt.imshow(batch_data[i].reshape(-1,ndf), cmap='gray', interpolation='none')
        plt.axis('off')
    plt.show()

optlr = 1e-4
optbeta1 = 0.3
IMGSZ = 64

input = torch.FloatTensor(BATCHSZ, nc, IMGSZ, IMGSZ).cuda()
noise = torch.FloatTensor(BATCHSZ, z_indim, 1, 1).cuda()
fixed_noise = torch.FloatTensor(BATCHSZ, z_indim, 1, 1).normal_(0, 1).cuda()
label = torch.FloatTensor(BATCHSZ).cuda()
real_label = 1
fake_label = 0
fixed_noise = Variable(fixed_noise)

#initialising the networks
D = Discriminator()
G = Generator()
D.apply(weights_init)
G.apply(weights_init)
D = D.cuda()
G = G.cuda()
print(D)
print(G)

#instantialing the optimizers
optimizerD = tchoptim.Adam(D.parameters(), lr=optlr, betas=(optbeta1, 0.9))
optimizerG = tchoptim.Adam(G.parameters(), lr=optlr, betas=(optbeta1, 0.9))

criterion = tchnn.BCELoss().cuda()

n_epoch = 200
n_iter = 1000
n_critic = 5

for eph in range(n_epoch):
    training_loss = 0
    for itr in range(n_iter):
        for i_critic in range(n_critic):
            #first training the D
            D.zero_grad()
            #train with real
            data = next(MNISTd)
            data = data.reshape(-1,nc,IMGSZ,IMGSZ)
            data = tch.from_numpy(data).cuda().float()
            real_cpu = data
            batch_size = real_cpu.size(0)
            real_cpu = real_cpu.cuda()
            input.resize_as_(real_cpu).copy_(real_cpu)
            label.resize_(batch_size).fill_(real_label)
            inputv = Variable(input)
            labelv = Variable(label)

            output = D(inputv)
            errD_real = criterion(output, labelv)
            errD_real.backward()
            D_x = output.data.mean()

            #train with fake
            noise.resize_(batch_size, z_indim, 1, 1).normal_(0, 1)
            noisev = Variable(noise)
            fake = G(noisev)
            labelv = Variable(label.fill_(fake_label))
            output = D(fake.detach())
            errD_fake = criterion(output, labelv)
            errD_fake.backward()
            D_G_z1 = output.data.mean()
            errD = errD_real + errD_fake
            optimizerD.step()
            #print(i_critic, itr, 'done')
        
        #traininig G
        G.zero_grad()
        noise.resize_(batch_size, z_indim, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        fake = G(noisev)
        labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
        output = D(fake)
        errG = criterion(output, labelv)
        errG.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()
        
        if itr%5 == 0:
            print('===> [{}/{}][{}/{}], Loss_D: {:.4f}, Loss_G: {:.4f}, D(x): {:.4f}, D(G(z)): {:.4f}|{:.4f}'.format(
                eph, n_epoch, itr, n_iter, errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
    #generation after this epoch
    print('######### generating samples after epoch: {} ##########'.format(eph))
    rand_inp = torch.FloatTensor(1000, z_indim, 1, 1).cuda()
    rand_inp.resize_(1000, z_indim, 1, 1).normal_(0, 1)
    V_rand_inp = Variable(rand_inp)
    genX = G(V_rand_inp)
    genX_data = genX.data.cpu().numpy()
    genX_datal = list(genX_data)
    genX_datal = random.sample(genX_datal, 64)
    genX_data = np.array(genX_datal)
    vis.images(genX_data, opts=dict(title='after epoch:{}, itr:{}'.format(eph,itr), 
    	caption='randomly chosen 64 generated samples after epoch: {}'.format(eph)),)
    #plotter64(genX_data)
    #save here
    tch.save(G.state_dict(), './models/unimodal/G_epoch{}.pth'.format(eph))
    tch.save(D.state_dict(), './models/unimodal/D_epoch{}.pth'.format(eph))

