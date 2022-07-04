## RHF: ResNet50-HDC-FPN-FasterRCNN-CMFD-LCMFD

## Model weighting file

* Link: https://pan.baidu.com/s/1oW3BopexHkJdsQlb79hsdw Extraction code: 2swh


## Dataset files CMFD & LCMFD

* Link: https://pan.baidu.com/s/1TLEdIfqfQXI-PT49Snv3aQ Extraction code: 1111


## Environment configuration.
* Python3.6/3.7/3.8
* Pytorch1.7.1
* pycocotools(Linux:``pip install pycocotools``; Windows:``pip install pycocotools-windows``)
* Partial code reference: https://github.com/WZMIAOMIAO/deep-learning-for-image-processing

## File structure.
* backbone: feature extraction network ResNet50
* backbone_hdc: ResNet50-HDC The hybrid inflated convolution rate integrated residual network of this paper
* network_files: Faster R-CNN network (including Fast R-CNN and modules such as RPN)
* train_utils: training validation related modules (including cocotools)
* my_dataset.py: custom dataset for reading VOC datasets
* train_mobilenet.py: use MobileNetV2 as the backbone for training
* train_resnet50_fpn.py: use resnet50 + FPN as backbone for training
* train_resnet50_hdc_fpn: train with resnet50-HDC + FPN as backbone
* train_multi_GPU.py: train with multiple GPUs
* predict.py: Simple prediction script to perform prediction tests using trained weights
* validation.py: validate/test the COCO metrics of the data using the trained weights, and generate record_mAP.txt file
* classes.json: pascal_voc tag file


## Pre-trained weights download address (download and put in backbone folder).
* MobileNetV2 backbone: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
* ResNet50+FPN backbone: https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
* Note that the downloaded pre-training weights should be renamed, e.g., the ``fasterrcnn_resnet50_fpn_coco.pth`` file is read in train_resnet50_fpn.py
  Not ``fasterrcnn_resnet50_fpn_coco-258fb6c6.pth``
 
 
## dataset

* LCMFD 
  * num of all_annotations: 8232
  * num of with_mask: 3229
  * num of poor_mask: 2813
  * num of none_mask: 2190
        
* CMFD:MaskDatasets_Augment 
  * num of all_annotations: 131422
  * num of with_mask: 53039
  * num of poor_mask: 47203
  * num of none_mask: 31180


## Training method
* Note that the dataset file name in *.py is changed to the dataset file name on your computer
* Modify the parameters in train_res50_fpn.py train_res50_hdc_fpn.py
    * Modify train_res50_fpn.py line 185 to replace data_path
    * Modify train_res50_hdc_fpn.py line 185 Replace data_path
* Make sure the backbone folder contains pre-trained model weights
* If you want to train mobilenetv2+fasterrcnn, use the train_mobilenet.py training script directly
* To train resnet50+fpn+fasterrcnn, use the train_resnet50_fpn.py training script directly
* To train resnet50-hdc+fpn+fasterrcnn, use the train_resnet50_hdc_fpn.py training script directly
* To train with multiple GPUs, use the ``python -m torch.distributed.launch --nproc_per_node=8 --use_env train_multi_GPU.py`` command, with the ``nproc_per_node`` parameter being the number of GPUs used
* If you want to specify which GPU devices to use you can prefix the command with ``CUDA_VISIBLE_DEVICES=0,3`` (e.g. I just want to use the 1st and 4th GPU devices in the device)
* ```CUDA_VISIBLE_DEVICES=0,3 python -m torch.distributed.launch --nproc_per_node=2 --use_env train_multi_GPU.py``

## Caution
* When using the training script, be careful to set '--data-path' (mask_root) to the **root directory** where you store the 'MaskDatasets_Augment' or 'MaskDatasets_NotAugment' folder
* Since Faster RCNN with FPN structure is very memory hungry, if the GPU memory is not enough (if the batch_size is less than 8), it is recommended to use the default norm_layer in the create_model function.
  If the GPU memory is not enough (if the batch_size is less than 8), it is recommended to use the default norm_layer in the create_model function.
* When using the prediction script, set 'train_weights' to your own generated weight path.
* When using the validation file, take care to make sure that your validation set or test set must contain targets for each class, and use it with only '--num-classes', '--data-path' and '--weights', and leave the rest of the code unchanged as much as possible