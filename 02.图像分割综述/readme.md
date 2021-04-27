# 图像分割综述



## 语义分割-Introduction



### Part 1 理论部分

![SemSeg_Index](assets/SemSeg_Index.png)

#### 1. 什么是图像分割

给一张图，把感兴趣的区域标出来，并判断它属于哪一类，或拥有何种语义

![SemSeg_PaperCounts.png](assets/SemSeg_PaperCounts.png)

图像分割在顶会中（CVPR为例）的份量、热度逐年增长。

![SemSeg_Acquire](assets/SemSeg_Acquire.png)

![SemSeg_Acquire2](assets/SemSeg_Acquire2.png)

学习这么课你将学习到什么

![SemSeg_Prerequisite](assets/SemSeg_Prerequisite.png)

学习这么课你需要提前学习什么



> The divel in the details
>
> 细节控



#### 课程大纲

![SemSeg_Index1](assets/SemSeg_Index1.png)

![SemSeg_Index2](assets/SemSeg_Index2.png)

![SemSeg_Index3](assets/SemSeg_Index3.png)



#### 2. 图像分割类型

![SemSeg_Category](assets/SemSeg_Category.png)

其中，全景分割相比较实例分割，就是再加上背景部分

![SemSeg_Category2](assets/SemSeg_Category2.png)

从上图能更明显理解到，全景分割实际上是 = semantic segmentation + instance segmentation



![SemSeg_Category3_VOS](assets/SemSeg_Category3_VOS.png)

Video Object Segmentation

![SemSeg_Category3_VIS](assets/SemSeg_Category3_VIS.png)

Video Instance Segmentation



![SemSeg_ApplicationScenario](assets/SemSeg_ApplicationScenario.png)

图像分割的应用场景



#### 3. 语义分割算法的基本概念

![SemSeg_BasicConception](assets/SemSeg_BasicConception.png)

根本目的：像素级分类（pixelwise classification）

![SemSeg_BasicConception_Workflow](assets/SemSeg_BasicConception_Workflow.png)

基本流程

![SemSeg_BasicConception_Metrics](assets/SemSeg_BasicConception_Metrics.png)

评价指标

- mAcc：mean Accuracy
- mIoU：mean Intersection-Over-Union

![SemSeg_BasicConception_Metrics_mAcc](assets/SemSeg_BasicConception_Metrics_mAcc.png)

对预测的结果逐像素对比，预测对了就 + 1，预测错了就 pass，然后得到的正确预测结果除以预测的像素数量，就是我们常说的 mAcc

![SemSeg_BasicConception_Metrics_mIoU](assets/SemSeg_BasicConception_Metrics_mIoU.png)

为什么 backward 不用 mIoU 用 loss？

实验上 loss 更好，评价上 mIoU 更全面

计算过程参考 [IoU计算][]



### Part 2 实践部分

![SemSeg_Recap](assets/SemSeg_Recap.png)

```bash
vim basic_model.py
```

基本模型实践

basic_model.py

```python
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import to_variable
from paddle.fluid.dygraph import Pool2D
from paddle.fluid.dygraph import Conv2D
import numpy as np
np.set_printoptions(precision=2)

class BasicModel(fluid.dygraph.Layer):
    def __init__(self, num_classes=59):
        super(BasicModel, self).__init__()
        self.pool = Pool2D(pool_size=2, pool_stride=2)
        self.conv = Conv2D(num_channels=3, num_filters=num_classes, filter_size=1)
        
    def forward(self, inputs):
        x = self.pool(inputs)
        x = fluid.layers.interpolate(x, out_shape=(inputs.shape[2:]))
        x = self.conv(x)
        
        return x
        

def main():
    place = paddle.fluid.CPUPlace()
    # place = paddle.fluid.CUDAPlace(0)
    with fluid.dygraph.guard(place):
        model = BasicModel(num_classes=59)
        model.eval() # model.train()
        # paddle 默认是 32bit 浮点型，整型会报错，所以要转换数据类型
        input_data = np.random.rand(1, 3, 8, 8).astype(np.float32)
        print('Input data shape:', input_data.shape)
        input_data = to_variable(input_data)
        output_data = model(input_data)
        output_data = outut_data.numpy()
        print('Output data shape:', output_data.shape)
        
    
if __name__=="__main__":
    main()
```

数据预处理实践

```bash
vim basic_dataloader.py
```

basic_dataloader.py

```python
import os
import random
import numpy as np
import cv2
import paddle.fluid as fluid

class Transform(object):
    def __init__(self, size=256):
        self.size = size
        
    def __call__(self, input, label):
        input = cv2.resize(input, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        # 用最近邻可以规避因为双线性插值造成的 label 结果不存在的情况，所以一定要使用最近邻
        label = cv2.resize(label, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        

class BasicDataLoader(object):
    def __init__(self,
                 image_folder,
                 image_list_file,
                 transform=None,
                 shuffle=True):
        self.image_folder = image_folder
        self.image_list_file = image_list_file
        self.transform = transform
        self.shuffle = shuffle
        
        self.data_list = self.read_list()
    
    def read_list(self):
        data_list = []
        with open(self.image_list_file) as infile:
            for line in infile:
                data_path = os.path.join(self.image_folder, line.split()[0])
                label_path = os.path.join(self.image_folder,line.split()[1])
                data_list.append((data_path, label_path))
        if self.shuffle:
        	random.shuffle(data_list)
        return data_list
    
    def preprocess(self, data, label):
        h, w, c = data.shape
        h_gt, w_gt = label.shape
        assert h==h_gt, "Error"
        assert w==w_gt, "Error"
        if self.transform:
            data, label = self.transform(data, label)
        label = label[:, :, np.newaxis]
        return data, label
    
    def __len__(self):
        return len(self.data_list)
    
    def __call__(self):
        for data_path, label_path in self.data_list:
            data = cv2.imread(data_path, cv2.IMREAD_COLOR)
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            label = cv2.imread(label_path, cv2.IMREAD_GRAY)
            print(data.shape, label.shape)
            data, label = self.preprocess( data, label)
            yield data, label
            
    
def main():
    batch_size = 5
    place = fluid.CUDAPlace(0)
    with fluid.dygraph.guard(place):
        transform = Transform(256)
        # create BasicDataloader instance
        basic_dataloader = BasicDataloader(
            image_folder='./path_to_img/',
            image_list_file='./path_to_img/list.txt',
            transform=transform,
            shuffle=True
        )
        
        # create fluid.io.DataLoader instance
        dataloader = fluid.io.DataLoader.from_generator(
            capacity=1,
            use_multiprocess=False
        )
        
        
        # set sample generator for fluid dataloader
        dataloader.set_sample_generator(
            basic_dataloader,
            batch_size=batch_size,
            places=place
        )
        
        num_epoch = 2
        for epoch in range(1, num_epoch+1):
            print(f'Epoch [{epoch}/{num_epoch}]:')
            for idx, (data, label) in enumerate(dataloader):
                print(f'Iter {idx}, Data shape: {data.shape}, Label shape:{label.shape}')

if __name__=='__main__':
    main()
```







[IoU计算]: https://blog.csdn.net/lingzhou33/article/details/87901365	"语义分割的评价指标——IoU"

