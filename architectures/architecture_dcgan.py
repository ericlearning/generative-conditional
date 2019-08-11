import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as SpectralNorm

class Nothing(nn.Module):
	def __init__(self):
		super(Nothing, self).__init__()
		
	def forward(self, x):
		return x

def get_norm(norm_type, size):
	if(norm_type == 'batchnorm'):
		return nn.BatchNorm2d(size)
	elif(norm_type == 'instancenorm'):
		return nn.InstanceNorm2d(size)

def get_activation(activation_type):
	if(activation_type == 'relu'):
		return nn.ReLU(inplace = True)
	elif(activation_type == 'leakyrelu'):
		return nn.LeakyReLU(0.2, inplace = True)
	elif(activation_type == 'elu'):
		return nn.ELU(inplace = True)
	elif(activation_type == 'selu'):
		return nn.SELU(inplace = True)
	elif(activation_type == 'prelu'):
		return nn.PReLU()
	elif(activation_type == 'tanh'):
		return nn.Tanh()
	elif(activation_type == None):
		return Nothing()

class ConvBlock(nn.Module):
	def __init__(self, ni, no, ks, stride, pad = None, use_bn = True, use_sn = False, use_pixelshuffle = False, norm_type = 'batchnorm', activation_type = 'leakyrelu', pad_type = 'Zero'):
		super(ConvBlock, self).__init__()
		self.use_bn = use_bn
		self.use_sn = use_sn
		self.use_pixelshuffle = use_pixelshuffle
		self.norm_type = norm_type
		self.pad_type = pad_type

		if(pad == None):
			pad = ks // 2 // stride

		ni_ = ni
		if(use_pixelshuffle):
			self.pixelshuffle = nn.PixelShuffle(2)
			ni_ = ni // 4
		
		if(pad_type == 'Zero'):
			self.conv = nn.Conv2d(ni_, no, ks, stride, pad, bias = False)
		else:
			self.conv = nn.Sequential(*[
				nn.ReflectionPad2d(pad),
				nn.Conv2d(ni_, no, ks, stride, 0, bias = False)
			])

		if(self.use_bn):
			self.bn = get_norm(norm_type, no)
		if(self.use_sn):
			self.conv = SpectralNorm(self.conv)

		self.act = get_activation(activation_type)

	def forward(self, x):
		out = x
		if(self.use_pixelshuffle):
			out = self.pixelshuffle(out)
		out = self.conv(out)
		if(self.use_bn):
			out = self.bn(out)
		out = self.act(out)
		return out

class DeConvBlock(nn.Module):
	def __init__(self, ni, no, ks, stride, pad = None, output_pad = 0, use_bn = True, use_sn = False, norm_type = 'batchnorm', activation_type = 'leakyrelu', pad_type = 'Zero'):
		super(DeConvBlock, self).__init__()
		self.use_bn = use_bn
		self.use_sn = use_sn
		self.norm_type = norm_type
		self.pad_type = pad_type

		if(pad is None):
			pad = ks // 2 // stride

		if(pad_type == 'Zero'):
			self.deconv = nn.ConvTranspose2d(ni, no, ks, stride, pad, output_padding = output_pad, bias = False)
		else:
			self.deconv = nn.Sequential(*[
				nn.ReflectionPad2d(pad),
				nn.ConvTranspose2d(ni, no, ks, stride, 0, output_padding = output_pad, bias = False)
			])

		if(self.use_bn):
			self.bn = get_norm(norm_type, no)
		if(self.use_sn):
			self.deconv = SpectralNorm(self.deconv)

		self.act = get_activation(activation_type)

	def forward(self, x):
		out = self.deconv(x)
		if(self.use_bn):
			out = self.bn(out)
		out = self.act(out)
		return out

class ConditionalBatchNorm(nn.Module):
	def __init__(self, n_classes, nc):
		super(ConditionalBatchNorm, self).__init__()
		self.nc = nc
		self.n_classes = n_classes
		self.bn = nn.BatchNorm2d(nc, affine = False)
		self.embed = nn.Embedding(n_classes, nc*2)
		self.embed.weights.data[:, :nc].normal_(0.0, 0.02)
		self.embed.weights.data[:, nc:].zero_()

	def forward(self, x, y):
		# x : (bs, nc, sz, sz)
		# y : (bs, n_classes)
		out_x = self.bn(x)
		out_y = self.embed(y)
		# out_y : (bs, nc * 2)
		gamma, beta = torch.chunk(out_y, 2, 1)
		gamma = gamma.view(-1, self.nc, 1, 1)
		beta = beta.view(-1, self.nc, 1, 1)
		# gamma : (bs, nc, 1, 1)
		# beta : (bs, nc, 1, 1)
		# out_x : (bs, nc, sz, sz)
		out = gamma * out_x + beta
		# out : (bs, nc, sz, sz)
		return out

# Conditional DCGAN Architectures

class Conditional_DCGAN_D(nn.Module):
	def __init__(self, sz, nc, n_classes, ndf = 64, use_sigmoid = True, use_bn = True, use_sn = False, norm_type = 'batchnorm', activation_type = 'leakyrelu'):
		super(Conditional_DCGAN_D, self).__init__()
		assert sz > 4, "Image size should be bigger than 4"
		assert sz & (sz-1) == 0, "Image size should be a power of 2"
		self.sz = sz
		self.nc = nc
		self.n_classes = n_classes
		self.ndf = ndf
		self.use_bn = use_bn
		self.use_sn = use_sn
		self.norm_type = norm_type

		cur_ndf = ndf
		layers = []
		self.first_conv = ConvBlock(self.nc, self.ndf, 4, 2, 1, use_bn = False, use_sn = self.use_sn, activation_type = activation_type)
		for i in range(int(math.log2(self.sz)) - 3):
			if(i == 0):
				layers.append(ConvBlock(cur_ndf+self.n_classes, cur_ndf * 2, 4, 2, 1, use_bn = self.use_bn, use_sn = self.use_sn, norm_type = norm_type, activation_type = activation_type))
			else:
				layers.append(ConvBlock(cur_ndf, cur_ndf * 2, 4, 2, 1, use_bn = self.use_bn, use_sn = self.use_sn, norm_type = norm_type, activation_type = activation_type))
			cur_ndf *= 2
		layers.append(ConvBlock(cur_ndf, 1, 4, 1, 0, use_bn = False, use_sn = self.use_sn, activation_type = None))

		self.main = nn.Sequential(*layers)
		self.sigmoid = nn.Sigmoid()
		self.use_sigmoid = use_sigmoid

		for m in self.modules():
			if(isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d)):
				m.weight.data.normal_(0.0, 0.02)
				if(m.bias is not None):
					m.bias.data.zero_()

	def forward(self, x, y):
		y = y.view(-1, self.n_classes, 1, 1)
		out = self.first_conv(x)
		out = torch.cat([out, y.repeat(1, 1, x.shape[2]//2, x.shape[3]//2)], dim = 1)
		out = self.main(out)
		if(self.use_sigmoid == True):
			out = self.sigmoid(out)
		return out

class Conditional_DCGAN_G(nn.Module):
	def __init__(self, sz, nz, nc, n_classes, ngf = 64, use_bn = True, use_sn = False, norm_type = 'batchnorm', activation_type = 'leakyrelu'):
		super(Conditional_DCGAN_G, self).__init__()
		self.sz = sz
		self.nz = nz
		self.nc = nc
		self.n_classes = n_classes
		self.ngf = ngf
		self.use_bn = use_bn
		self.use_sn = use_sn
		self.norm_type = norm_type

		cur_ngf = ngf * self.sz // 8
		layers = [DeConvBlock(self.nz + self.n_classes, cur_ngf, 4, 1, 0, use_bn = self.use_bn, use_sn = self.use_sn, norm_type = self.norm_type, activation_type = activation_type)]
		for i in range(int(math.log2(self.sz)) - 3):
			layers.append(DeConvBlock(cur_ngf, cur_ngf // 2, 4, 2, 1, use_bn = self.use_bn, use_sn = self.use_sn, norm_type = self.norm_type, activation_type = activation_type))
			cur_ngf = cur_ngf // 2
		layers.append(DeConvBlock(self.ngf, self.nc, 4, 2, 1, use_bn = False, use_sn = self.use_sn, activation_type = 'tanh'))

		self.main = nn.Sequential(*layers)
		for m in self.modules():
			if(isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d)):
				m.weight.data.normal_(0.0, 0.02)
				if(m.bias is not None):
					m.bias.data.zero_()
					
	def forward(self, x, y):
		y = y.view(-1, self.n_classes, 1, 1)
		out = self.main(torch.cat([x, y], dim = 1))
		return out