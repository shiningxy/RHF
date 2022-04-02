# Faster R-CNN

## 模型权重文件

* 链接：https://pan.baidu.com/s/1oW3BopexHkJdsQlb79hsdw 提取码：2swh


## 数据集文件 CMFD & Light-CMFD

* 链接：https://pan.baidu.com/s/1wXpmu3sI2pNiaRMzc-e6-A 提取码：yywr 


## 环境配置：
* Python3.6/3.7/3.8
* Pytorch1.7.1
* pycocotools(Linux:```pip install pycocotools```; Windows:```pip install pycocotools-windows```)
* 部分代码参考：https://github.com/WZMIAOMIAO/deep-learning-for-image-processing

## 文件结构：
* backbone: 特征提取网络 ResNet50
* backbone_hdc: ResNet50-HDC 本文的混合膨胀卷积率集成残差网络
* network_files: Faster R-CNN网络（包括Fast R-CNN以及RPN等模块）
* train_utils: 训练验证相关模块（包括cocotools）
* my_dataset.py: 自定义dataset用于读取VOC数据集
* train_mobilenet.py: 以MobileNetV2做为backbone进行训练
* train_resnet50_fpn.py: 以resnet50 + FPN作为backbone进行训练
* train_resnet50_hdc_fpn: 以resnet50-HDC + FPN作为backbone进行训练
* train_multi_GPU.py: 使用多GPU训练
* predict.py: 简易的预测脚本，使用训练好的权重进行预测测试
* validation.py: 利用训练好的权重验证/测试数据的COCO指标，并生成record_mAP.txt文件
* classes.json: pascal_voc标签文件


## 预训练权重下载地址（下载后放入backbone文件夹中）：
* MobileNetV2 backbone: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
* ResNet50+FPN backbone: https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
* 注意，下载的预训练权重记得要重命名，比如在train_resnet50_fpn.py中读取的是```fasterrcnn_resnet50_fpn_coco.pth```文件，
  不是```fasterrcnn_resnet50_fpn_coco-258fb6c6.pth```
 
 
## 数据集

* MaskDatasets_NotAugment 
    * 数据增强前的数据集
        * num of all_annotations:  8232
        * num of with_mask:  3229
        * num of poor_mask:  2813
        * num of none_mask:  2190
        
* MaskDatasets_Augment 
    * 数据增强后的数据集
        * num of all_annotations:  131422
        * num of with_mask:  53039
        * num of poor_mask:  47203
        * num of none_mask:  31180


## 训练方法
* 注意修改*.py中的数据集文件名为自己电脑中的数据集文件名
* 修改train_res50_fpn.py train_res50_hdc_fpn.py中的参数
    * 修改train_res50_fpn.py 185行 替换data_path
    * 修改train_res50_hdc_fpn.py 185行 替换data_path
* 确保backbone文件夹内有预训练模型权重
* 若要训练mobilenetv2+fasterrcnn，直接使用train_mobilenet.py训练脚本
* 若要训练resnet50+fpn+fasterrcnn，直接使用train_resnet50_fpn.py训练脚本
* 若要训练resnet50-hdc+fpn+fasterrcnn，直接使用train_resnet50_hdc_fpn.py训练脚本
* 若要使用多GPU训练，使用```python -m torch.distributed.launch --nproc_per_node=8 --use_env train_multi_GPU.py```指令,```nproc_per_node```参数为使用GPU数量
* 如果想指定使用哪些GPU设备可在指令前加上```CUDA_VISIBLE_DEVICES=0,3```(例如我只要使用设备中的第1块和第4块GPU设备)
* ```CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 --use_env train_multi_GPU.py```

## 注意事项
* 在使用训练脚本时，注意要将'--data-path'(mask_root)设置为自己存放'MaskDatasets_Augment'或'MaskDatasets_NotAugment'文件夹所在的**根目录**
* 由于带有FPN结构的Faster RCNN很吃显存，如果GPU的显存不够(如果batch_size小于8的话)建议在create_model函数中使用默认的norm_layer，
  即不传递norm_layer变量，默认去使用FrozenBatchNorm2d(即不会去更新参数的bn层),使用中发现效果也很好。
* 在使用预测脚本时，要将'train_weights'设置为你自己生成的权重路径。
* 使用validation文件时，注意确保你的验证集或者测试集中必须包含每个类别的目标，并且使用时只需要修改'--num-classes'、'--data-path'和'--weights'即可，其他代码尽量不要改动
