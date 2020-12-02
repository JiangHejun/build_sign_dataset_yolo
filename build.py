'''
Description: build single signatrue dataset, lable num: 0, <object-class> <x_center> <y_center> <width> <height>
Author: Hejun Jiang
Date: 2020-11-27 10:29:38
LastEditTime: 2020-12-02 12:09:37
LastEditors: Hejun Jiang
Version: v0.0.1
Contact: jianghejun@hccl.ioa.ac.cn
Corporation: hccl
'''
import os
import cv2  # h,w,c
import shutil
import random
import argparse
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--finetune', action='store_true', help='show a done picture for finetune, to ensure the parameters is satisfied')
parser.add_argument('--sign-size-ratio', type=list, default=[0.05, 0.08], help='the ratio of sign height size for background')
parser.add_argument('--sign-rotate', type=float, default=0.2, help='the ratio of rotate sign in background')
parser.add_argument('--img-add-noise', type=float, default=0.2, help='the ratio of image add noise')
parser.add_argument('--sp-noise-ratio', type=list, default=[0.005, 0.02], help='the parameter of sp noise')
parser.add_argument('--gauss-noise-ratio', type=list, default=[0.04, 0.07], help='the parameter of gauss noise')
parser.add_argument('--signnum', type=list, default=[1, 10], help='the number scale of sign in background')
parser.add_argument('--datasetdir', type=str, default='./dataset', help='the dir of dataset; background, signature must in here')
parser.add_argument('--trainnum', type=int, default=2048, help='the number of train images for building')
parser.add_argument('--valnum', type=int, default=512, help='the number of val images for building')
parser.add_argument('--testnum', type=int, default=1024, help='the number of test images for building')
parser.add_argument('--projectname', type=str, default='sign', help='the name of project')
conf = parser.parse_args()

imageType = ['jpg', 'png', 'jpeg']
rotateType = [Image.ROTATE_90,Image.ROTATE_180,Image.ROTATE_270] # sign 进行旋转
datasetdir = os.path.abspath(conf.datasetdir)
backDir = os.path.join(datasetdir,'background')
signDir = os.path.join(datasetdir, 'signature')
yamlPath = os.path.join(datasetdir, 'coco_'+conf.projectname+'.yaml')

imagesDir = os.path.join(datasetdir, 'images')
if os.path.isdir(imagesDir):
    shutil.rmtree(imagesDir)
trainImDir = os.path.join(imagesDir,'train_sign')
valImDir = os.path.join(imagesDir,'val_sign')
testImDir = os.path.join(imagesDir, 'test_sign')
os.makedirs(trainImDir)
os.makedirs(valImDir)
os.makedirs(testImDir)

labelsDir = os.path.join(datasetdir, 'labels')
if os.path.isdir(labelsDir):
    shutil.rmtree(labelsDir)
trainLaDir = os.path.join(labelsDir,'train_sign')
valLaDir = os.path.join(labelsDir,'val_sign')
testLaDir = os.path.join(labelsDir, 'test_sign')
os.makedirs(trainLaDir)
os.makedirs(valLaDir)
os.makedirs(testLaDir)

assert conf.sign_size_ratio[1] > conf.sign_size_ratio[0], 'sign_size_ratio[1] must bigger than sign_size_ratio[0]'
assert conf.sp_noise_ratio[1]>conf.sp_noise_ratio[0], 'sp_noise_ratio[1] must bigger than sp_noise_ratio[0]'
assert conf.gauss_noise_ratio[1]>conf.gauss_noise_ratio[0], 'gauss_noise_ratio[1] must bigger than gauss_noise_ratio[0]'
assert conf.signnum[1]>conf.signnum[0], 'signnum[1] must bigger than signnum[0]'
print('*******************build sign dataset for yolov5, please use --finetune first*******************')
print('finetune:', conf.finetune)
print('sign_size_ratio:', conf.sign_size_ratio)
print('sign_rotate:', conf.sign_rotate)
print('img_add_noise:', conf.img_add_noise)
print('sp_noise_ratio:', conf.sp_noise_ratio)
print('gauss_noise_ratio:', conf.gauss_noise_ratio)
print('signnum:', conf.signnum)
print('trainnum:', conf.trainnum)
print('valnum:', conf.valnum)
print('testnum:', conf.testnum)
print('projectname:', conf.projectname)
print('imageType:', imageType)
print('rotateType:', rotateType)
print('datasetdir:', datasetdir)
print('backDir:', backDir)
print('signDir:', signDir)
print('yamlPath:', yamlPath)
print('trainImDir:', trainImDir)
print('valImDir:', valImDir)
print('testImDir:', testImDir)
print('trainLaDir:', trainLaDir)
print('valLaDir:', valLaDir)
print('testLaDir:', testLaDir)

def getPathList(dir):
    pathlis, objlis = [], set()
    for root, dirs, files in os.walk(dir):
        for name in files:
            if name.split('.')[-1] in imageType:        
                pathlis.append(os.path.join(root, name))
                objlis.add(os.path.basename(root))
        
    return pathlis, list(objlis)

def saveYaml(objlis):
    f = open(yamlPath, 'w', encoding='utf-8')
    f.write('# please revise parameters train/val/test if their dir is moved\n')
    f.write('train: '+trainImDir+'\n')
    f.write('val: '+valImDir+'\n')
    f.write('test: '+testImDir+'\n\n')
    f.write('nc: ' + str(len(objlis)) + '\n')
    nstr = ''
    for name in objlis:
        nstr += '\'' + name + '\'' + ','
    if len(nstr) > 0: # 去除最后一个','
        nstr = nstr[:-1]
    f.write('names: [' + nstr + ']\n')
    f.close()

def gaussValue(scale):
    while True:
        r = random.gauss(scale[0], scale[1])
        if r >= scale[0] and r <= scale[1]:
            return r

def getbox(bimg, rsimg, addedboxs):
    hlen, wlen = rsimg.shape[0], rsimg.shape[1]
    for i in range(1000):
        h = random.randint(0, bimg.shape[0] - hlen)
        w = random.randint(0, bimg.shape[1] - wlen)
        ch = h +  hlen // 2 #中心点位置
        cw = w + wlen // 2
        isin = False
        for box in addedboxs:
            if abs(2*(box[0] - ch)) < box[2] + h and abs(2*(box[1] - cw)) < box[3] + w:
                isin = True
                break
        if not isin:
            return [h,w,ch,cw,hlen,wlen]
    return [] # rand 1000次未出来，代表box差不多占满了

def sp_noise(image): # 椒盐
    prob1 = random.uniform(0.005, 0.02)  # prob*2
    prob2 = 1 - prob1
    output = np.random.random(image.shape)
    image[output < prob1] = 0
    image[output > prob2] = 255
    return image


def gauss_noise(image): # 高斯
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(0, random.uniform(0.04, 0.07), image.shape)
    out = image + noise
    out[out < 0] = 0
    out[out > 1] = 1
    out = np.uint8(out * 255)
    return out


def noiseRand(image):
    addnoise = [sp_noise, gauss_noise]
    rand = random.randint(0, 1)
    img = image
    if random.random()<conf.img_add_noise:
        img = random.choice(addnoise)(image)
    return img

def build(backList, signList, num, imgDir, lableDir, objlis):
    print('imgDir:',imgDir,'num:',num)
    for i in range(num):
        if (i + 1) % 100 == 0:
            print('      builded:', i+1)
        bimg = cv2.imread(random.choice(backList))
        slis = random.choices(signList, k=random.randint(conf.signnum[0], conf.signnum[1]))

        addedboxs = []  # 已添加的boxsize，避免之后的重复,[ch, cw, hlen, wlen]中心位置和其长宽
        f = open(os.path.join(lableDir, conf.projectname+"_"+str(i)+'.txt'),'w',encoding ='utf-8')
        for file in slis:
            simg = cv2.imread(file)  # 需保证签名图中白底
            r = gaussValue(conf.sign_size_ratio)  # 缩放
            rsimg = cv2.resize(simg, (int(bimg.shape[0] * r * simg.shape[1] / simg.shape[0]), int(bimg.shape[0] * r)))

            if random.random() < conf.sign_rotate: # 随机旋转
                roimg = Image.fromarray(cv2.cvtColor(rsimg, cv2.COLOR_BGR2RGB)).transpose(random.choice(rotateType))
                rsimg = cv2.cvtColor(np.asarray(roimg), cv2.COLOR_RGB2BGR)
            
            box = getbox(bimg, rsimg, addedboxs)
            if len(box) == 0:
                print('box is full', file)
                continue  # 代表差不多满了，这次放弃覆盖
            addedboxs.append(box[2:])

            roi = bimg[box[0] : rsimg.shape[0] + box[0], box[1] : rsimg.shape[1] + box[1]] # 抠图
            rsimggray = cv2.cvtColor(rsimg, cv2.COLOR_BGR2GRAY)
            bsize = max(rsimggray.shape[0], rsimggray.shape[1]) #越大杂点越少
            if bsize % 2 == 0:
                bsize += 1
            c = min(rsimggray.shape[0], rsimggray.shape[1]) # 越大越去除杂点
            mask = cv2.adaptiveThreshold(rsimggray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,bsize, c) # 黑字
            mask_inv = cv2.bitwise_not(mask)  # 颠倒黑白，白字
            
            rsimgbg = cv2.bitwise_and(roi, roi, mask=mask)  #当mask!=0,roi和roi按位与;当mask==0，为0;黑字
            rsimgfg = cv2.bitwise_and(rsimg, rsimg, mask=mask_inv)  # 当mask!=0,rsimg和rsimg按位与;当mask==0，为0；彩（黑）字
            bimg[box[0] : rsimg.shape[0] + box[0], box[1] : rsimg.shape[1] + box[1]] = cv2.add(rsimgbg, rsimgfg) #彩（黑）字覆盖黑字，在背景图中

            dirname = os.path.basename(os.path.dirname(file)) # 获取obj名
            f.write('%d %.6f %.6f %.6f %.6f\n'%(objlis.index(dirname),box[3]/bimg.shape[1],box[2]/bimg.shape[0], rsimg.shape[1]/bimg.shape[1],rsimg.shape[0]/bimg.shape[0]))
        f.close()
        bimg = noiseRand(bimg)
        if conf.finetune:
            cv2.imshow('bimg', bimg)
            cv2.waitKey(0)
            exit(0)
        cv2.imwrite(os.path.join(imgDir, conf.projectname + "_" + str(i) + '.jpg'), bimg)
        
    

if __name__ == '__main__':
    backList, _ = getPathList(backDir)
    signList, objlis = getPathList(signDir)
    saveYaml(objlis)
    build(backList, signList, conf.trainnum, trainImDir, trainLaDir, objlis)
    build(backList, signList, conf.valnum, valImDir, valLaDir, objlis)
    build(backList, signList, conf.testnum, testImDir, testLaDir, objlis)
    print('build dataset done')