# DPT for Object Detection

Here is our code for ImageNet classification. Please check [our paper](https://arxiv.org/abs/2107.14467) for detailed information.

## Instructions

### Preparations

First, install pytorch as for classification.
```bash
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
pip install timm==0.3.2
```

We develop our method under environment `mmcv==1.2.7` and `mmdet==2.8.0`. We recommand you [this document](https://github.com/open-mmlab/mmdetection/blob/v2.8.0/docs/get_started.md) for detailed instructions.

### Evaluation

To evaluate RetinaNet on COCO val2017 with 8 gpus run:
```
./dist_test.sh /path/to/config/file /path/to/checkpoint_file 8 --eval bbox
```

For example, to evaluate RetinaNet with DPT-Tiny:

```
./dist_test.sh configs/retinanet_dpt_t_fpn_1x_coco.py pretrained/detection/retinanet_dpt_t_1x.pth 8 --eval bbox
```


To evaluate Mask R-CNN on COCO val2017 with 8 gpus run:

```
./dist_test.sh /path/to/config/file /path/to/checkpoint_file 8 --eval bbox segm
```

For example, to evaluate Mask R-CNN with DPT-Tiny:

```
./dist_test.sh configs/mask_rcnn_dpt_t_fpn_1x_coco.py pretrained/detection/mrcnn_dpt_t_1x.pth 8 --eval bbox segm
```

### Training

Train with certain config file:

```
dist_train.sh /path/to/config/file $NUM_GPUS
```


For example, to train DPT-Small + Mask R-CNN on COCO train2017 for 12 epochs with 8 gpus:

```
dist_train.sh configs/mask_rcnn_dpt_s_fpn_1x_coco.py 8
```


## Results and Models
### RetinaNet Results

| Method     | #Params (M) | Schedule |  mAP | AP50 | AP75 |  APs |  APm |  APl |
|------------|:-----------:|:--------:|:----:|:----:|:----:|:----:|:----:|:----:|
| DPT-Tiny   |    24.9     |    1x    | 39.5 | 60.4 | 41.8 | 23.7 | 43.2 | 52.2 |
| DPT-Tiny   |    24.9     |    MS+3x | 41.2 | 62.0 | 44.0 | 25.7 | 44.6 | 53.9 |
| DPT-Small  |    36.1     |    1x    | 42.5 | 63.6 | 45.3 | 26.2 | 45.7 | 56.9 |
| DPT-Small  |    36.1     |    MS+3x | 43.3 | 64.0 | 46.5 | 27.8 | 46.3 | 58.5 |
| DPT-Medium |    55.9     |    1x    | 43.3 | 64.6 | 45.9 | 27.2 | 46.7 | 58.6 |
| DPT-Medium |    55.9     |    MS+3x | 43.7 | 64.6 | 46.4 | 27.2 | 47.0 | 58.4 |

### Mask R-CNN Results

| Method     | #Params (M) | Schedule | box mAP | box AP50 | box AP75 | mask mAP | mask AP50 | mask AP75 |
|------------|:-----------:|:--------:|:-------:|:--------:|:--------:|:--------:|:---------:|:---------:|
| DPT-Tiny   |    34.8     |    1x    |   40.2  |   62.8   |   43.8   |   37.7   |   59.8    |   40.4    |
| DPT-Tiny   |    34.8     |    MS+3x |   42.2  |   64.4   |   46.1   |   39.4   |   61.5    |   42.3    |
| DPT-Small  |    46.1     |    1x    |   43.1  |   65.7   |   47.2   |   39.9   |   62.9    |   43.0    |
| DPT-Small  |    46.1     |    MS+3x |   44.4  |   66.5   |   48.9   |   41.0   |   63.6    |   44.2    |
| DPT-Medium |    65.8     |    1x    |   43.8  |   66.2   |   48.3   |   40.3   |   63.1    |   43.4    |
| DPT-Medium |    65.8     |    MS+3x |   44.3  |   65.6   |   48.8   |   40.7   |   63.1    |   44.1    |

### BaiduNetDist Link

You can obtain the ImageNet1k pre-trained model from [BaiduNetdisk](https://pan.baidu.com/s/19nJXoOAK_mljV4BPx1sUSQ). Password for extract is **DPTs**.
(Google Drive Link will be provided soon.)
