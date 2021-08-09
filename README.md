# DPT
--------

This repo is the official implementation of **DPT: Deformable Patch-based Transformer for Visual Recognition (ACM MM2021)**. We provide code and models for the following tasks:

> **Image Classification**: Detailed instruction and information see [classification/README.md](classification/README.md).

> **Object Detection**: Detailed instruction and information see [detection/README.md](detection/README.md).

The papar has been relased on [[Arxiv](https://arxiv.org/abs/2107.14467)].

## Introduction

Deformable Patch (DePatch) is a plug-and-play module. It learns to adaptively split the images input patches with different positions and scales in a data-driven way, rather than using predefined fixed patches. In this way, our method can well preserve the semantics in patches.

In this repository, code and models for a Deformable Patch-based Transformer (DPT) are provided. As this field is developing rapidly, we are willing to see our DePatch applied to some other latest architectures and promote further research.

## Main Results

### Image Classification

Training commands and pretrained models are provided >>> [here](classification) <<<.

| Method     | #Params (M) | FLOPs(G) | Acc@1 |
|------------|:-----------:|:--------:|:-----:|
| DPT-Tiny   |    15.2     |   2.1    | 77.4  |
| DPT-Small  |    26.4     |   4.0    | 81.0  |
| DPT-Medium |    46.1     |   6.9    | 81.9  |

### Object Detection
Training command and detailed results are provided >>> [here](detection) <<<.

## Citation
```
@inproceedings{chenDPT21,
  title = {DPT: Deformable Patch-based Transformer for Visual Recognition},
  author = {Zhiyang Chen and Yousong Zhu and Chaoyang Zhao and Guosheng Hu and Wei Zeng and Jinqiao Wang and Ming Tang},
  booktitle={Proceedings of the ACM International Conference on Multimedia},
  year={2021}
}
```

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## Acknowledgement
Our implementation is mainly based on [PVT](https://github.com/whai362/PVT). The CUDA operator is borrowed from [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR). You may refer these repositories for further information.
