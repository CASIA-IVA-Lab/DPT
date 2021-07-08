# DPT
--------
This repo is the official implementation of **DPT: Deformable Patch-based Transformer for Visual Recognition (ACM MM2021)**. The paper will be released soon. We provide code and models for the following tasks:

> *Image Classification*: Instructions not ready.

> *Object Detection*: Instructions not ready.

## Introduction

Deformable Patch (DePatch) is a plug-and-play module. It learns to adaptively split the images input patches with different positions and scales in a data-driven way, rather than using predefined fixed patches. In this way, our method can well preserve the semantics in patches.

In this repository, code and models for a Deformable Patch-based Transformer (DPT) are provided. As this field is developing rapidly, we are willing to see our DePatch applied to some other latest architectures and promote further research.

## Main Results

### Image Classification
| Method     | #Params (M) | FLOPs(G) | Acc@1 | Model |
|------------|:-----------:|:--------:|:-----:|:-----:|
| DPT-Tiny   |    15.2     |   2.1    | 77.4  |       |
| DPT-Small  |    26.4     |   4.0    | 81.0  |       |
| DPT-Medium |    46.1     |   6.9    | 81.9  |       |

### Object Detection
Coming soon.

## Citation
Citation will be provided once the paper is released.

## License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

## Acknowledgement
Our implementation is mainly based on [PVT](https://github.com/whai362/PVT). The CUDA operator is borrowed from [Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR). You may refer these repositories for further information.

## Original Content (It will be removed when README.md is fullfilled.)

论文标题及链接

项目摘要

安装、训练（评测）

测试

模型（表格，并提供下载链接）

license

citing(bibtex)

等等，仅供参考，具体可以自行修改样式

# 附：
Markdown语法说明：https://www.appinn.com/markdown/

Markdown在线测试工具：http://mahua.jser.me/
