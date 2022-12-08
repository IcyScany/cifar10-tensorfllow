# 使用说明：
## 一、配置环境：
    python：3.7.11
    tensorflow: 1.15.0-gpu版本

## 代码运行要求：
### 1 数据设置 
    1.使用cifar10二进制数据集，在main.py文件中代码第16行 'datapath=xxxx' 处修改为自己所存储的数据路径。

### 2 GPU设置
该程序使用tensorflow-gpu版本， 在运行时可设置GPU使用情况： 

    1.在main.py文件中代码第11行 'os.environ["CUDA_VISIBLE_DEVICES"] = 'xxx'' 处设置使用的GPU。
    2.在main.py文件中代码第11行 'config.gpu_options.per_process_gpu_memory_fraction = xxx' 设置需要占用GPU的显存大小。

## 二、网络训练批正则化及梯度裁剪参数设置
该程序是使用神经网络对cifar10数据集进行分类，其中网络参数设置中有对网络进行Batch_Normal以及梯度裁剪操作
### 1. Batch_Normal批正则化操作
    若需要对网络设置批正则化操作，则在需在main.py程序main函数194行处设置 'batch_noraml' 为True

### 2. 梯度裁剪操作
    若需要对网络进行梯度裁剪操作， 则在需在main.py程序main函数197行处设置 'gvs_capped' 为True


