import torch
import os
from random import shuffle
import random
import numpy as np
class MyDataset(torch.utils.data.Dataset):#需要继承torch.utils.data.Dataset
    def __init__(self, foloder='../data/point', mode='train'):
        #对继承自父类的属性进行初始化(好像没有这句也可以？？)
        super(MyDataset,self).__init__()
        # TODO
        #1、初始化一些参数和函数，方便在__getitem__函数中调用。
        #2、制作__getitem__函数所要用到的图片和对应标签的list。
        #也就是在这个模块里，我们所做的工作就是初始化该类的一些基本参数。
        random.seed(0)
        self.path_save = []
        path = os.path.join(foloder, mode)
        for label in os.listdir(path):
            for img_path in os.listdir(os.path.join(path, label)):
                self.path_save.append([os.path.join(path, label, img_path), int(label)-1])
            
        # print(self.path_save[0])
        shuffle(self.path_save)
        # print(self.path_save[0])
        # print(len(self.path_save))
        if mode == 'train':
            self.path_save = self.path_save[:int(len(self.path_save)*0.8)]
        elif mode == 'val':
            self.path_save = self.path_save[int(len(self.path_save)*0.8):]
        pass
    def normalize(self, image):
        mean = np.mean(image)
        std = np.std(image)
        # print(std)
        # if std == 0:
        #     return
        image = (image - mean) / std
        return image, std
    def __getitem__(self, index):
        # TODO
        #1、根据list从文件中读取一个数据（例如，使用numpy.fromfile，PIL.Image.open）。
        #2、预处理数据（例如torchvision.Transform）。
        #3、返回数据对（例如图像和标签）。
        #这里需要注意的是，这步所处理的是index所对应的一个样本。
        img = np.load(self.path_save[index][0])
        # img = self.normalize(img)
        img /= 65536
        img, std = self.normalize(img)
        label = self.path_save[index][1]
        if not -1<label<10:
            print(label)
        # print(self.path_save[index][0])
        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.int8), self.path_save[index][0], std
        
    def __len__(self):
        #返回数据集大小
        return len(self.path_save)
        # print(len(self.path_save))
        # return 1280
