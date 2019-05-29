# -*- coding: utf-8 -*-
"""
Created on Wed May 29 16:58:33 2019

@author: Marcus
"""

import mxnet as mx
import numpy as np
import mxnet.ndarray as nd
import cv2, time, scipy
from matplotlib import pyplot as plt

class RandIter(mx.io.DataIter):
    def __init__(self, batch_size, ndim):
        self.batch_size = batch_size
        self.ndim = ndim
        self.provide_data = [('rand', (batch_size, ndim, 1, 1))]
        self.provide_label = []

    def iter_next(self):
        return True

    def getdata(self):
        #Returns random numbers from a gaussian (normal) distribution
        #with mean=0 and standard deviation = 1
        return [mx.random.normal(0, 1.0, shape=(self.batch_size, self.ndim, 1, 1))]
    
def getData(file_path):
    sperm_d = scipy.io.loadmat(file_path)['cnn_data']
    pre_data = []
    normal_data = []
    for i in sperm_d:
        for j in i[0]:
            pre_data.append(np.float64(j[0]))
            if j[1][0] == 3:
                normal_data.append(np.float64(j[0]))
    pre_data = np.asarray([cv2.resize(pre_data,(64,64)) for pre_data in pre_data])
    pre_data = pre_data.astype(np.float32, copy=False)/(255.0/2) - 1.0
    
    normal_data = np.asarray([cv2.resize( normal_data,(64,64)) for  normal_data in  normal_data])
    normal_data = normal_data.astype(np.float32, copy=False)/(255.0/2) - 1.0
#    plt.imshow(pre_data[0])
    pre_data = nd.array(pre_data).expand_dims(axis= 1)
    pre_data = nd.tile(pre_data,(1,3,1,1))
    
    normal_data = nd.array(normal_data).expand_dims(axis= 1)
    normal_data = nd.tile(normal_data,(1,3,1,1))
        
    
    return  pre_data.asnumpy(), normal_data.asnumpy()   

def fillBuf(buf, num_images, img, shape):
    width = buf.shape[0]/shape[1]
    height = buf.shape[1]/shape[0]
    img_width = int(num_images%width)*shape[0]
    img_hight = int(num_images/height)*shape[1]
    buf[img_hight:img_hight+shape[1], img_width:img_width+shape[0], :] = img

def visualize(fake, real):
    
    fake = fake.transpose((0, 2, 3, 1))
    
    fake = np.clip((fake+1.0)*(255.0/2.0), 0, 255).astype(np.uint8)
    
    real = real.transpose((0, 2, 3, 1))
    real = np.clip((real+1.0)*(255.0/2.0), 0, 255).astype(np.uint8)
    
    n = np.ceil(np.sqrt(fake.shape[0]))
    fbuff = np.zeros((int(n*fake.shape[1]), int(n*fake.shape[2]), int(fake.shape[3])), dtype=np.uint8)
    for i, img in enumerate(fake):
        fillBuf(fbuff, i, img, fake.shape[1:3])
    rbuff = np.zeros((int(n*real.shape[1]), int(n*real.shape[2]), int(real.shape[3])), dtype=np.uint8)
    for i, img in enumerate(real):
        fillBuf(rbuff, i, img, real.shape[1:3])
        
    fig = plt.figure()
    ax1 = fig.add_subplot(2,2,1)
    ax1.imshow(fbuff)
    ax2 = fig.add_subplot(2,2,2)
    ax2.imshow(rbuff)
    plt.show()
def vis(fake, real):
    
    fake = fake.transpose((0, 2, 3, 1))
    
    fake = np.clip((fake+1.0)*(255.0/2.0), 0, 255).astype(np.uint8)
    
    real = real.transpose((0, 2, 3, 1))
    real = np.clip((real+1.0)*(255.0/2.0), 0, 255).astype(np.uint8)

    
    n = np.ceil(np.sqrt(fake.shape[0]))
    fbuff = np.zeros((int(n*fake.shape[1]), int(n*fake.shape[2]), int(fake.shape[3])), dtype=np.uint8)
    for i, img in enumerate(fake):
        fillBuf(fbuff, i, img, fake.shape[1:3])
    rbuff = np.zeros((int(n*real.shape[1]), int(n*real.shape[2]), int(real.shape[3])), dtype=np.uint8)
    for i, img in enumerate(real):
        fillBuf(rbuff, i, img, real.shape[1:3])

    plt.figure(1)
    plt.imshow(fbuff)
    plt.savefig('generator_total')
    plt.figure(2)
    plt.imshow(rbuff)
    plt.savefig('Input image')

X_a,X = getData('D:/lagBear/SEMEN/finally_data/cnn_data.mat')

batch_size = 64
image_iter = mx.io.NDArrayIter(X, batch_size=batch_size)

Z = 100
rand_iter = RandIter(batch_size, Z)
no_bias = True
fix_gamma = True
epsilon = 1e-5 + 1e-12        
#=============G net=============

rand = mx.sym.Variable('rand')

g1 = mx.sym.Deconvolution(rand, name='g1', kernel=(4,4), num_filter=1024, no_bias=no_bias)
gbn1 = mx.sym.BatchNorm(g1, name='gbn1', fix_gamma=fix_gamma, eps=epsilon)
gact1 = mx.sym.Activation(gbn1, name='gact1', act_type='relu')

g2 = mx.sym.Deconvolution(gact1, name='g2', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=512, no_bias=no_bias)
gbn2 = mx.sym.BatchNorm(g2, name='gbn2', fix_gamma=fix_gamma, eps=epsilon)
gact2 = mx.sym.Activation(gbn2, name='gact2', act_type='relu')

g3 = mx.sym.Deconvolution(gact2, name='g3', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=256, no_bias=no_bias)
gbn3 = mx.sym.BatchNorm(g3, name='gbn3', fix_gamma=fix_gamma, eps=epsilon)
gact3 = mx.sym.Activation(gbn3, name='gact3', act_type='relu')

g4 = mx.sym.Deconvolution(gact3, name='g4', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=128, no_bias=no_bias)
gbn4 = mx.sym.BatchNorm(g4, name='gbn4', fix_gamma=fix_gamma, eps=epsilon)
gact4 = mx.sym.Activation(gbn4, name='gact4', act_type='relu')

g5 = mx.sym.Deconvolution(gact4, name='g5', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=3, no_bias=no_bias)
generatorSymbol = mx.sym.Activation(g5, name='gact5', act_type='tanh')

#=============D net=============


data = mx.sym.Variable('data')

d1 = mx.sym.Convolution(data, name='d1', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=128, no_bias=no_bias)
dact1 = mx.sym.LeakyReLU(d1, name='dact1', act_type='leaky', slope=0.2)

d2 = mx.sym.Convolution(dact1, name='d2', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=256, no_bias=no_bias)
dbn2 = mx.sym.BatchNorm(d2, name='dbn2', fix_gamma=fix_gamma, eps=epsilon)
dact2 = mx.sym.LeakyReLU(dbn2, name='dact2', act_type='leaky', slope=0.2)

d3 = mx.sym.Convolution(dact2, name='d3', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=512, no_bias=no_bias)
dbn3 = mx.sym.BatchNorm(d3, name='dbn3', fix_gamma=fix_gamma, eps=epsilon)
dact3 = mx.sym.LeakyReLU(dbn3, name='dact3', act_type='leaky', slope=0.2)

d4 = mx.sym.Convolution(dact3, name='d4', kernel=(4,4), stride=(2,2), pad=(1,1), num_filter=1024, no_bias=no_bias)
dbn4 = mx.sym.BatchNorm(d4, name='dbn4', fix_gamma=fix_gamma, eps=epsilon)
dact4 = mx.sym.LeakyReLU(dbn4, name='dact4', act_type='leaky', slope=0.2)

d5 = mx.sym.Convolution(dact4, name='d5', kernel=(4,4), num_filter=1, no_bias=no_bias)
d5 = mx.sym.Flatten(d5)

label = mx.sym.Variable('label')
discriminatorSymbol = mx.sym.LogisticRegressionOutput(data=d5, label=label, name='dloss')

# 超參數
sigma = 0.02
lr = 0.0002
beta1 = 0.5
ctx = mx.gpu(0) 

#=============G Module=============
generator = mx.mod.Module(symbol=generatorSymbol, data_names=('rand',), label_names=None, context=ctx)
generator.bind(data_shapes=rand_iter.provide_data)
generator.init_params(initializer=mx.init.Normal(sigma))
generator.init_optimizer(
    optimizer='adam',
    optimizer_params={
        'learning_rate': lr,
        'beta1': beta1,
    })
mods = [generator]

# =============D Module=============
discriminator = mx.mod.Module(symbol=discriminatorSymbol, data_names=('data',), label_names=('label',), context=ctx)
discriminator.bind(data_shapes=image_iter.provide_data,
          label_shapes=[('label', (batch_size,))],
          inputs_need_grad=True)
discriminator.init_params(initializer=mx.init.Normal(sigma))
discriminator.init_optimizer(
    optimizer='adam',
    optimizer_params={
        'learning_rate': lr,
        'beta1': beta1,
    })
mods.append(discriminator)

# =============start train===============
start = time.time()
print('Training...') 
for epoch in range(10000):
    image_iter.reset()
    for i, batch in enumerate(image_iter):
        rbatch = rand_iter.next()
        generator.forward(rbatch, is_train=True)
        outG = generator.get_outputs()
        label = mx.nd.zeros((batch_size,), ctx=ctx)
        
        discriminator.forward(mx.io.DataBatch(outG, [label]), is_train=True)
        discriminator.backward()
        gradD = [[grad.copyto(grad.context) for grad in grads] for grads in discriminator._exec_group.grad_arrays]

        label[:] = 1
        
        batch.label = [label]
        
        discriminator.forward(batch, is_train=True)
        discriminator.backward()
        for gradsr, gradsf in zip(discriminator._exec_group.grad_arrays, gradD):
            for gradr, gradf in zip(gradsr, gradsf):
                gradr += gradf
        
        discriminator.update()

        discriminator.forward(mx.io.DataBatch(outG, [label]), is_train=True)
        discriminator.backward()
        
        diffD = discriminator.get_input_grads()
        generator.backward(diffD)
        generator.update()

        
        i += 1
        if i % 4 == 0:
            print('epoch:', epoch, 'iter:', i)
            print
            print("   From generator:        From MNIST:")

            visualize(outG[0].asnumpy(), batch.data[0].asnumpy())
            
            
plt.figure(1)            
fa = outG[0][0][0].asnumpy()
fb = batch.data[0][0][0].asnumpy()
plt.imshow(fa,cmap = 'gray')
plt.figure(2)
plt.imshow(fb,cmap = 'gray')

vis(outG[0].asnumpy(), batch.data[0].asnumpy())
end = time.time()
print(end-start)            