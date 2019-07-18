import os
import torch
import torch.nn as nn
from dataset import Dataset
from architectures.architecture_dcgan import Conditional_DCGAN_D, Conditional_DCGAN_G
from trainers_advanced.trainer import Trainer
from utils import save, load

dir_name = 'data/mnist'
basic_types = 'MNIST'

lr_D, lr_G, bs = 0.0002, 0.0002, 128
sz, nc, nz, n_classes, ngf, ndf = 64, 3, 100, 10, 64, 64
use_sigmoid, spectral_norm, attention_layer = False, True, 256

data = Dataset(dir_name, basic_types)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
netD = Conditional_DCGAN_D(sz, nc, n_classes, ndf, use_sigmoid).to(device)
netG = Conditional_DCGAN_G(sz, nz, nc, n_classes, ngf).to(device)

trn_dl = data.get_loader(sz, bs)

trainer = Trainer('SGAN', netD, netG, n_classes, device, trn_dl, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = None, use_gradient_penalty = False, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')
trainer = Trainer('LSGAN', netD, netG, n_classes, device, trn_dl, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = None, use_gradient_penalty = False, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')
trainer = Trainer('HINGEGAN', netD, netG, n_classes, device, trn_dl, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = None, use_gradient_penalty = False, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')
trainer = Trainer('WGAN', netD, netG, n_classes, device, trn_dl, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = 0.01, use_gradient_penalty = False, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')
trainer = Trainer('WGAN', netD, netG, n_classes, device, trn_dl, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = None, use_gradient_penalty = 10, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')

trainer = Trainer('RASGAN', netD, netG, n_classes, device, trn_dl, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = None, use_gradient_penalty = False, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')
trainer = Trainer('RALSGAN', netD, netG, n_classes, device, trn_dl, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = None, use_gradient_penalty = False, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')
trainer = Trainer('RAHINGEGAN', netD, netG, n_classes, device, trn_dl, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = None, use_gradient_penalty = False, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')

trainer = Trainer('QPGAN', netD, netG, n_classes, device, trn_dl, lr_D = lr_D, lr_G = lr_G, resample = True, weight_clip = None, use_gradient_penalty = False, loss_interval = 150, image_interval = 300, save_img_dir = 'saved_imges')

trainer.train(5)
save('saved/cur_state.state', netD, netG, trainer.optimizerD, trainer.optimizerG)
