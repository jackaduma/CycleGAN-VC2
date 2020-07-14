# **CycleGAN-VC2**

## **Voice Conversion by CycleGAN**

This code is implemented for [CycleGAN-VC2: Improved CycleGAN-based Non-parallel Voice Conversion](https://arxiv.org/abs/1904.04631])

------

## [**Project Page**](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/cyclegan-vc2/index.html)

------

![network](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/cyclegan-vc2/images/network.png "network")


------
## TODO LIST
- [x] Dataset
  - [ ] VC
  - [x] Chinese Male Speakers (S0913 from AISHELL-Speech & GaoXiaoSong: a Chinese star)
- [x] Usage
  - [x] Training
  - [ ] Infer
- [ ] Demo

------

## **Usage**

**requirement** 

```bash
pip install -r requirements.txt
```

**preprocess**

```python
python preprocess_training.py
```
is short for

```python
python preprocess_training.py --train_A_dir ./data/S0913/ --train_B_dir ./data/gaoxiaosong/ --cache_folder ./cache/
```


**train** 
```python
python train.py
```

is short for

```python
python train.py --logf0s_normalization ./cache/logf0s_normalization.npz --mcep_normalization ./cache/mcep_normalization.npz --coded_sps_A_norm ./cache/coded_sps_A_norm.pickle --coded_sps_B_norm ./cache/coded_sps_B_norm.pickle --model_checkpoint ./model_checkpoint/ --resume_training_at ./model_checkpoint/_CycleGAN_CheckPoint --validation_A_dir ./data/S0913/ --output_A_dir ./converted_sound/S0913 --validation_B_dir ./data/gaoxiaosong/ --output_B_dir ./converted_sound/gaoxiaosong/
```

------

## **pretrained model**

------

## **Demo**

example

S0913

<audio id="audio" controls="" preload="none">
<source id="mp3" src="./data/S0913/BAC009S0913W0351.wav">
</audio>

GaoXiaoSong

<audio id="audio" controls="" preload="none">
<source id="mp3" src="./data/gaoxiaosong/gaoxiaosong_1.wav">
</audio>

Converted (S0913 -> GaoXiaoSong)

<audio id="audio" controls="" preload="none">
<source id="mp3" src="./converted_sound/S0913/BAC009S0913W0351.wav">
</audio>