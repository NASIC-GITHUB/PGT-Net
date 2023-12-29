# PGT-Net

### Description

- PGT-Net by Yu-Ting Li(109062529).
- Thesis: PGT-Net: Progressive Guided Multi-tasks Neural Network for Wet Fingerprint Denoising
- Author: 109062529 Yu-Ting Li, Department of Computer Science, National Tsing Hua University

### Environment

- python=3.8.8
- tensorflow-gpu=2.6.0

### Self-defined variables setup

#### Training

- Modifiy global variables in main.py line 34 - 36, these variable defines the saving path of trained model, and the file name of trained model.
- ft-lightnoised dataset

```python
# training & validation dataset path
dataset_root = '../dataset/training_validation/ft_lightnoised_157x36.npz'
# filename of model
save_name = '0911_epoch_ft'
# path of model
result_path = './result/'
```

- fw9395 dataset

```python
# training & validation dataset path
dataset_root = '../dataset/training_validation/fw9395_176x36.npz'
# filename of model
save_name = '0911_epoch_ft'
# path of model
result_path = './result/'
```

#### Testing

- Modifiy global variables in main.py line 39 - 54, these variable defines the path of test dataset, and the saving path of output image.
- fT-lightnoised dataset

```python
# binary testset ground truth
b_normal_img_path = '../dataset/testing/ft_lightnoised_dataset/crop_black_area_include_binary/binary/y_test'
# non-binary testset input
nb_test_img_path = '../dataset/testing/ft_lightnoised_dataset/crop_black_area_include_binary/non_binary/x_test'
# non-binary testset ground truth
nb_normal_img_path = '../dataset/testing/ft_lightnoised_dataset/crop_black_area_include_binary/non_binary/y_test'
# path of binary testing output
b_test_result_path = './0911_testing/binary'
# path of non-binary testing output
nb_test_result_path = './0911_testing/non_binary'
# path of output statistic file
csv_path = "./0911_testing/ft_lightnoised_157x36.csv"
# path of output learning rate statistic file (debug)
lr_csv_path = "./0911_testing/ft_lightnoised_157x36_lr.csv"
```

- fw9395 dataset

```python
# binary testset ground truth
b_normal_img_path = '../dataset/testing/FW9395_dataset/testset/binary/y_test'
# non-binary testset input
nb_test_img_path = '../dataset/testing/FW9395_dataset/testset/non_binary/x_test'
# non-binary testset ground truth
nb_normal_img_path = '../dataset/testing/FW9395_dataset/testset/non_binary/y_test'
# path of binary testing output
b_test_result_path = './0911_testing/binary'
# path of non-binary testing output
nb_test_result_path = './0911_testing/non_binary'
# path of output statistic file
csv_path = "./0911_testing/FW9395.csv"
# path of output learning rate statistic file (debug)
lr_csv_path = "./0911_testing/FW9395_lr.csv"
```

##### Global variables

- Modifiy global variables in main.py line 58 - 67
  If you want to use a pretrained model, set variable "load_weights" to the pretrained weight file path, otherwise just assign a empty string.

```python
# pretrained weight needed
load_weights = "./<path_to_your_pretrained_weight_file>"
## the pretrained weight of fw9395 dataset
load_weights = "./result/xxx.h5"
# pretrained weight not needed
load_weights = ""
```

- If you want to train a new model

```python
training = True
```

- If you want to test a new model

```python
testing = True
```

- If you want to change the model version, change variable "model_version"

```python
# yuting's origin PGT-Net-Block-84
model_version = 0
# change non_binary branch: stage24*binary_output, stage60+BFM
model_version = 1
# change binary branch: stage48+BFM
model_version = 2
# change both branch: stage48+BFM, stage24*binary_output, stage60+BFM
model_version = 3
# only change to add at the end of each branch: stage48+BFM, stage60+BFM
model_version = 4
```

### How to use

```
$ python main.py
```

- The testing output images and the statistic of images will be saved to the path you set.
- The statistic csv file contains the MSE, SSIM, PSNR between the output testing images & their ground truth.
