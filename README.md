# PatchGenCN

TensorFlow Implementation for the paper:

**[Patchwise Generative ConvNet: Training Energy-Based Models
from a Single Natural Image for Internal Learning](https://openaccess.thecvf.com/content/CVPR2021/html/Zheng_Patchwise_Generative_ConvNet_Training_Energy-Based_Models_From_a_Single_Natural_CVPR_2021_paper.html)**  
In CVPR 2021 <span style="color:red;font-weight:bold">(Oral)</span>

[**Paper**](https://openaccess.thecvf.com/content/CVPR2021/papers/Zheng_Patchwise_Generative_ConvNet_Training_Energy-Based_Models_From_a_Single_Natural_CVPR_2021_paper.pdf) | [**Project**](https://zilongzheng.github.io/PatchGenCN/)

![](assets/example.png)

## Getting Started
This codebase is tested using Ubuntu 16.04, Python 3.5 and a single NVIDIA RTX 2080 GPU. Similar configurations are preferred.

### Installation
- Clone this repo:
```bash
git clone https://github.com/zilongzheng/PatchGenCN.git
cd PatchGenCN
```
- Install requirements
    - Tensorflow 1.14+

### Train
```bash
python train.py --datapath <path to image>
```

## Citation
If you use this code for your research, please cite our paper.
```bibtex
@InProceedings{zheng2021patchgencn,
    author    = {Zheng, Zilong and Xie, Jianwen and Li, Ping},
    title     = {Patchwise Generative ConvNet: Training Energy-Based Models From a Single Natural Image for Internal Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {2961-2970}
}
```