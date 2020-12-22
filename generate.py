'''
Description: 
Author: Hejun Jiang
Date: 2020-12-22 15:21:59
LastEditTime: 2020-12-22 15:28:13
LastEditors: Hejun Jiang
Version: v0.0.1
Contact: jianghejun@hccl.ioa.ac.cn
Corporation: hccl
'''
import os
import torch
import model
import numpy as np
from torchvision.utils import save_image


def GetDevice(cuda, dev):
    if cuda:
        device = torch.device("cuda:0")
        print('using gpu', dev)
    else:
        device = torch.device("cpu")
        print('using cpu')
    return device


def generate(dev, modelpath, gennum, savedir):
    if dev != '' and dev != 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = dev
    cuda = True if torch.cuda.is_available() and dev != 'cpu' else False
    imgh, latent_dim, channels = 320, 100, 1

    generator = model.Generator(imgh, latent_dim, channels)  # h, dim, c
    print(generator)
    pth = torch.load(modelpath, map_location=GetDevice(cuda, dev))
    generator.load_state_dict(pth['generator_model_state_dict'])
    generator.eval()  # 设置为评估模式；如果是resume，则设置为train模式
    print('generator load parameters done')

    for i in range(gennum):
        if (i + 1) % 100 == 0:
            print('generated fake images num', i+1)
        z = torch.FloatTensor(np.random.normal(0, 1, (1, latent_dim)))  # 输出的值赋在shape里,(batchsize, latent_dim)
        gen_imgs = generator(z)  # [gennum, 1, 320, 320] b, c, h, w
        save_image(gen_imgs.data, os.path.join(savedir, 'fake_{}.png'.format(i)), nrow=1, normalize=True)
    print('generate done')
