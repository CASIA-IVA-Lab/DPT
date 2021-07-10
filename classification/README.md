# DPT for Image Classification
-----
Here is our code for ImageNet classification. Please check our paper (coming soon) for detailed information.

## Instructions

### Environment

We develop our model under `cuda 10.1`, `pytorch 1.7.1` and `timm 0.3.2`. Pytorch with other versions may also work. We advise you to prepare your environment with `conda`.
```bash
conda install pytorch==1.7.1 torchvision==0.8.2 cudatoolkit=10.1 -c pytorch
pip install timm==0.3.2
```

You may clone our repo and compile the provided operator.
```bash
git clone https://github.com/CASIA-IVA-Lab/DPT.git
cd ./ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```

### Data Preparation

We follow the conventional way to prepare the ImangeNet dataset.

The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val/` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

### Evaluation

To evaluate a pretrained model on ImageNet val on a single gpus:

```bash
python -m torch.distributed.launch --nproc_per_node 1 --use_env main.py --eval --model $MODEL_NAME --data-path $DATA_PATH --resume $CKPT_PATH
```

Or with multiple gpus:

```bash
python -m torch.distributed.launch --nproc_per_node $NUM_GPUS --use_env main.py --eval --dist-eval --model $MODEL_NAME --data-path $DATA_PATH --resume $CKPT_PATH
```

For example, use 8 gpu to test our pretrained DPT-Small model.
```bash
python -m torch.distributed.launch --nproc_per_node 8 --use_env main.py --eval --dist-eval --model dpt_tiny --data-path $DATA_PATH --resume dpt_tiny.pth
```
which should give
```
* Acc@1 80.954 Acc@5 95.388 loss 0.846
Accuracy of the network on the 50000 test images: 81.0%
```


### Training

To train DPT-Small on ImageNet on a single node with 8 gpus for 300 epochs run:

```bash
MODEL_NAME=dpt_small
DATA_PATH=/path/to/imagenet
OUTPUT_PATH=/path/to/output

python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py\
 --model $MODEL_NAME --batch-size 128 --dist-eval --test_interval 5\
 --data-path $DATA_PATH --output_dir $OUTPUT_PATH
```

## Model Zoo

| Method     | #Params (M) | FLOPs(G) | Acc@1 | Model |
|------------|:-----------:|:--------:|:-----:|:-----:|
| DPT-Tiny   |    15.2     |   2.1    | 77.4  |       |
| DPT-Small  |    26.4     |   4.0    | 81.0  |       |
| DPT-Medium |    46.1     |   6.9    | 81.9  |       |

Our model will be released soon.

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.
