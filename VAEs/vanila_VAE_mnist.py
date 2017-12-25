import torch as tch
import torch.nn.functional as F
import torch.nn as tchnn
import torch.autograd as tchagd
import torch.optim as tchoptim
from torch.autograd import Variable
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os, shutil, errno
import random
import visdom

#make dir for models if not already present
try:
	os.makedirs('./models')
except OSError as exception:
	if exception.errno != errno.EEXIST:
		raise


#params for the system
e_inp_dim = 28*28
e_out_dim = 100 #z, latent variable
d_inp_dim = 100
d_out_dim = 28*28
n_hidden = 512
n_iter = 10000
BATCHSZ = 64*2

#visdom client
vis = visdom.Visdom(port=7777)

#encoder network (X -> z)
class Encoder(tchnn.Module):
	def __init__(self):
		super(Encoder, self).__init__()
		self.L1 = tchnn.Linear(e_inp_dim, n_hidden)
		#self.L2 = tchnn.Linear(n_hidden, n_hidden)
		self.Ou = tchnn.Linear(n_hidden, e_out_dim) #for mu and sigma 
		self.L3 = tchnn.Linear(e_out_dim, e_out_dim)#for mu
		self.L4 = tchnn.Linear(e_out_dim, e_out_dim)#for sigma
	
	def forward(self, x):
		x = F.relu(self.L1(x))
		#x = F.relu(self.L2(x))
		x = F.relu(self.Ou(x))
		#mu = x[:, :e_out_dim]
		#sig = x[:, e_out_dim:]
		mu = self.L3(x)
		sig = self.L4(x)
		#need to make sure that sigmas are positive
		#sig = F.softplus(sig)
		return mu, sig
		
	def name(self):
		return 'Encoder'

class Decoder(tchnn.Module):
	def __init__(self):
		super(Decoder, self).__init__()
		self.L1 = tchnn.Linear(d_inp_dim, n_hidden)
		#self.L2 = tchnn.Linear(n_hidden, n_hidden)
		self.Ou = tchnn.Linear(n_hidden, d_out_dim)
	
	def forward(self, x):
		x = F.relu(self.L1(x))
		#x = F.relu(self.L2(x))
		x = F.sigmoid(self.Ou(x))
		return x
	
	def name(self):
		return 'Decoder'

#xavier weight initialisation, suggested to be good
def wt_init(m):
	classname = m.__class__.__name__
	if classname.find('Linear') != -1:
		my_xavier(m.weight.data.cpu())
		m.bias.data.fill_(0)

def my_xavier(W):
	size = W.size()
	in_dim = size[0]
	xavier_stddev = 1. / np.sqrt(in_dim / 2.)
	return Variable(tch.randn(*size) * xavier_stddev, requires_grad=True).cuda()

#data loading and related stuff
MNISTX_train = np.load('../../gmmGAN/MNISTX_train.npy')
def MNIST_gen(X, BATCHSZ):
	X = X.reshape(-1,784) #serialize images
	while(True):
		databatch = random.sample(list(X), BATCHSZ)
		databatch = np.array(databatch)
		yield databatch
MNISTd = MNIST_gen(MNISTX_train, BATCHSZ)

#mnist batch plotter
def plotter(batch_data):
	#batch_data = batch_data.numpy()
	n = batch_data.shape[0]
	for i in range(n):
		plt.subplot(8,8,i+1)
		plt.imshow(batch_data[i].reshape(-1,28), cmap='gray', interpolation='none')
		plt.axis('off')
	plt.show()

#instantiating the model and optimizer
E = Encoder().cuda()
D = Decoder().cuda()
E = E.double()
D = D.double()
E.apply(wt_init)
D.apply(wt_init)
print(E)
print(D)
optim = tchoptim.Adam(list(E.parameters())+ list(D.parameters()), lr=1e-3)

def sample_z(mu, sig):
	e = Variable(tch.randn(BATCHSZ, e_out_dim).cuda().double())
	r = mu + (e*tch.exp(sig/2))
	return r

#training
for itr in range(n_iter):
	X = next(MNISTd)
	X = Variable(tch.from_numpy(X).cuda())
	#through the encoder
	E.zero_grad()
	z_mu, z_log_sig = E(X)
	z = sample_z(z_mu, z_log_sig)
	#through the decoder
	z = z.cuda()
	
	D.zero_grad()
	d_out = D(z)
	#loss
	recon_loss = F.binary_cross_entropy(d_out, X)
	KL_div = tch.mean(0.5 * tch.sum(tch.exp(z_log_sig) + z_mu**2 - 1. - z_log_sig, 1))
	KL_div /= BATCHSZ*28*28 # <- learnt a lesson, constansdo matter, kind of like units
	loss = recon_loss + KL_div
	
	loss.backward()
	optim.step()
	
	#print results sometimes
	if itr%100 == 0:
		print('randomly chosen 64 generated samples after itr: {}'.format(itr))
		d = d_out.data.cpu().numpy()[0:64,:]
		d = d.reshape(-1,1,28,28)
		d1 = list(d)
		d1 = random.sample(d1, 64)
		d = np.array(d1)
		vis.images(d, 
			opts=dict(title='after itr:{}'.format(itr), caption='randomly chosen 64 generated samples after itr: {}'.format(itr)),)
	if itr%1000 == 999:
		tch.save(E, './models/E_{}.pth'.format(itr))
		tch.save(D, './models/D_{}.pth'.format(itr))
	
