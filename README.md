# DFIN 
## We will put all the codes up after the paper is officially published
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
To train our PFRRNetwith wpdc, wpdc68 and graph_structure Loss, use
```
cd training
bash train_dfin.sh
```
## Futher Information
If you have any problems with the code, please list the problems you encountered in the issue area, and I will reply you soon. Thanks for the baseline work [3DDFA](https://github.com/cleardusk/3DDFA).
