# DFIN 
## Our paper has officially published
Title: Deformable Feature Interaction Network and Graph Structure Reasoning for 3D Dense Alignment and Face Reconstruction
IEEE Address:  (https://ieeexplore.ieee.org/abstract/document/10191307)

Welcome to read our paper,we would love to hear from you!




## Installation
First you have to make sure that you have all dependencies in place.
You can create an anaconda environment called *DFIN* using

```
conda env create -n DFIN python=3.6 ## recommended python=3.6+
conda activate DFIN
sudo pip3 install torch torchvision 
sudo pip3 install numpy scipy matplotlib
sudo pip3 install dlib
sudo pip3 install opencv-python
sudo pip3 install cython
sudo pip3 install mmcv-full
```
## Data
| Data      | Download Link | Description     |
| :---:        |    :----:   |          :---: |
| train.configs      | [BaiduYun](https://pan.baidu.com/s/1ozZVs26-xE49sF7nystrKQ#list/path=%2F), 217M       |The directory containing 3DMM params and filelists of training dataset   |
| train_aug_120x120.zip   | [BaiduYun](https://pan.baidu.com/s/19QNGst2E1pRKL7Dtx_L1MA)        | The cropped images of augmentation training dataset   |
| test.data.zip   | [BaiduYun](https://pan.baidu.com/s/1DTVGCG5k0jjjhOc8GcSLOw )       | The cropped images of AFLW and ALFW-2000-3D testset      |
## Generation
First, compile the extension modules.
```
cd utils/cython
python3 setup.py build_ext -i
```
To generate results using a trained model, use
```
python3 main.py -f samples/test.jpg 
```
## Evaluation
To eval our DFIN , use
```
python benchmark.py
```
## Training
To train our DFIN with wpdc, wpdc68 and graph_structure Loss, use
```
cd training
bash train_dfin.sh
```
## Futher Information
If you have any problems with the code, please list the problems you encountered in the issue area, and I will reply you soon. Thanks for the baseline work [3DDFA](https://github.com/cleardusk/3DDFA).
