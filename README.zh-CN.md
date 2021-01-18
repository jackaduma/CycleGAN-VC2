# **CycleGAN-VC2-PyTorch**

[![standard-readme compliant](https://img.shields.io/badge/readme%20style-standard-brightgreen.svg?style=flat-square)](https://github.com/jackaduma/CycleGAN-VC2)

[**中文说明**](./README.zh-CN.md) | [**English**](./README.md)

本项目使用**PyTorch**复现论文：[CycleGAN-VC2: Improved CycleGAN-based Non-parallel Voice Conversion](https://arxiv.org/abs/1904.04631]), 在**音色转换/声音克隆**方面非常优秀的算法模型.

- [x] 数据集
  - [ ] VC
  - [x] 中文男性说话人(S0913 from [AISHELL-Speech](https://openslr.org/33/) & [GaoXiaoSong: a Chinese star](https://en.wikipedia.org/wiki/Gao_Xiaosong))
- [x] 用法
  - [x] 训练
  - [x] Example 
- [ ] Demo

------

## **CycleGAN-VC2**

### [**论文项目主页**](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/cyclegan-vc2/index.html)


To advance the research on non-parallel VC, we propose CycleGAN-VC2, which is an improved version of CycleGAN-VC incorporating three new techniques: an improved objective (two-step adversarial losses), improved generator (2-1-2D CNN), and improved discriminator (Patch GAN).


![network](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/cyclegan-vc2/images/network.png "network")

------

**本项目包括:** 

1. [模型代码](model_tf.py) ，复现论文中的算法模型.
2. [语音预处理](preprocess_training.py)，对[训练数据](data)进行处理.
3. [训练代码](train.py)，训练模型.
4. [Examples of Voice Conversion](converted_sound/) - 模型训练后的转换样本。

------

## **内容列表**

- [**CycleGAN-VC2-PyTorch**](#cyclegan-vc2-pytorch)
  - [**CycleGAN-VC2**](#cyclegan-vc2)
    - [**论文项目主页**](#论文项目主页)
  - [**内容列表**](#内容列表)
  - [**依赖**](#依赖)
  - [**用法**](#用法)
    - [**预处理**](#预处理)
    - [**训练**](#训练)
  - [**预训练模型**](#预训练模型)
  - [**Demo**](#demo)
  - [**引用**](#引用)
  - [**捐赠**](#捐赠) 
  - [**License**](#license)
  
------



## **依赖** 

```bash
pip install -r requirements.txt
```
## **用法**

### **预处理**

```python
python preprocess_training.py
```
自定义参数执行：

```python
python preprocess_training.py --train_A_dir ./data/S0913/ --train_B_dir ./data/gaoxiaosong/ --cache_folder ./cache/
```


### **训练** 
```python
python train.py
```

自定义参数执行：

```python
python train.py --logf0s_normalization ./cache/logf0s_normalization.npz --mcep_normalization ./cache/mcep_normalization.npz --coded_sps_A_norm ./cache/coded_sps_A_norm.pickle --coded_sps_B_norm ./cache/coded_sps_B_norm.pickle --model_checkpoint ./model_checkpoint/ --resume_training_at ./model_checkpoint/_CycleGAN_CheckPoint --validation_A_dir ./data/S0913/ --output_A_dir ./converted_sound/S0913 --validation_B_dir ./data/gaoxiaosong/ --output_B_dir ./converted_sound/gaoxiaosong/
```

------

## **预训练模型**

a pretrained model which converted between S0913 and GaoXiaoSong

download from [Google Drive](https://drive.google.com/file/d/1iamizL98NWIPw4pw0nF-7b6eoBJrxEfj/view?usp=sharing) <735MB>

------

## **Demo**

使用预训练模型转换的样本:


**说话人A**: [S0913(./data/S0913/BAC009S0913W0351.wav)](https://drive.google.com/file/d/14zU1mI8QtoBwb8cHkNdZiPmXI6Mj6pVW/view?usp=sharing)

**说话人B**: [GaoXiaoSong(./data/gaoxiaosong/gaoxiaosong_1.wav)](https://drive.google.com/file/d/1s0ip6JwnWmYoWFcEQBwVIIdHJSqPThR3/view?usp=sharing)



**说话人A的语音转换为说话人B的音色**: [Converted from S0913 to GaoXiaoSong (./converted_sound/S0913/BAC009S0913W0351.wav)](https://drive.google.com/file/d/1S4vSNGM-T0RTo_aclxRgIPkUJ7NEqmjU/view?usp=sharing)

------

## **引用**
1. **CycleGAN-VC2: Improved CycleGAN-based Non-parallel Voice Conversion**. [Paper](https://arxiv.org/abs/1904.04631), [Project](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/cyclegan-vc2/index.html)
2. Parallel-Data-Free Voice Conversion Using Cycle-Consistent Adversarial Networks. [Paper](https://arxiv.org/abs/1711.11293), [Project](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/cyclegan-vc/)
3. Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. [Paper](https://arxiv.org/abs/1703.10593), [Project](https://junyanz.github.io/CycleGAN/), [Code](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
4. Image-to-Image Translation with Conditional Adversarial Nets. [Paper](https://arxiv.org/abs/1611.07004), [Project](https://phillipi.github.io/pix2pix/), [Code](https://github.com/phillipi/pix2pix)

------

## 捐赠
If this project help you reduce time to develop, you can give me a cup of coffee :) 

AliPay(支付宝)
<div align="center">
	<img src="./misc/ali_pay.png" alt="ali_pay" width="400" />
</div>

WechatPay(微信)
<div align="center">
    <img src="./misc/wechat_pay.png" alt="wechat_pay" width="400" />
</div>

[![paypal](https://www.paypalobjects.com/en_US/i/btn/btn_donateCC_LG.gif)](https://paypal.me/jackaduma?locale.x=zh_XC)


------

## **License**

[MIT](LICENSE) © Kun