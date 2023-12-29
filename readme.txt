########## step 1 - environment setup #########
    python 3.8.8
    tensorflow-gpu 2.6.0

########## step 2 - self-defined training variables setup #########
    Modifiy global variables in main.py line 34 - 36, these variable defines the saving path of trained model, and the file name of trained model.

    If you want to use ft-lightnoised dataset, the variables can be set as below example:
        dataset_root = '../dataset/training_validation/ft_lightnoised_157x36.npz'								# training & validation dataset path
        save_name = '0911_epoch_ft'																				# filename of model
        result_path = './result/'																				# path of model

    If you want to use fw9395 dataset, the variables can be set as below example:
        dataset_root = '../dataset/training_validation/fw9395_176x36.npz'								        # training & validation dataset path
        save_name = '0911_epoch_ft'																				# filename of model
        result_path = './result/'																				# path of model     

########## step 3 - self-defined testing variables setup #########
    Modifiy global variables in main.py line 39 - 54, these variable defines the path of test dataset, and the saving path of output image.

    If you want to use FT-lightnoised dataset, the variables can be set as below example:
        b_normal_img_path = '../dataset/testing/ft_lightnoised_dataset/crop_black_area_include_binary/binary/y_test'										# binary testset ground truth
        nb_test_img_path = '../dataset/testing/ft_lightnoised_dataset/crop_black_area_include_binary/non_binary/x_test'										# non-binary testset input
        nb_normal_img_path = '../dataset/testing/ft_lightnoised_dataset/crop_black_area_include_binary/non_binary/y_test'									# non-binary testset ground truth

        b_test_result_path = './0911_testing/binary'							                                                                            # path of binary testing output
        nb_test_result_path = './0911_testing/non_binary'						                                                                            # path of non-binary testing output

        csv_path = "./0911_testing/ft_lightnoised_157x36.csv"										                                                        # path of output statistic file
        lr_csv_path = "./0911_testing/ft_lightnoised_157x36_lr.csv"									                                                        # path of output learning rate statistic file (debug)

    If you want to use fw9395 dataset, the variables can be set as below example:
        b_normal_img_path = '../dataset/testing/FW9395_dataset/testset/binary/y_test'										                                # binary testset ground truth
        nb_test_img_path = '../dataset/testing/FW9395_dataset/testset/non_binary/x_test'										                            # non-binary testset input
        nb_normal_img_path = '../dataset/testing/FW9395_dataset/testset/non_binary/y_test'									                                # non-binary testset ground truth

        b_test_result_path = './0911_testing/binary'							                                                                            # path of binary testing output
        nb_test_result_path = './0911_testing/non_binary'						                                                                            # path of non-binary testing output

        csv_path = "./0911_testing/FW9395.csv"										                                                        # path of output statistic file
        lr_csv_path = "./0911_testing/FW9395_lr.csv"									                                                        # path of output learning rate statistic file (debug)

########## step 4 - self-defined testing variables setup #########
Modifiy global variables in main.py line 58 - 67
If you want to use a pretrained model, set variable "load_weights" to the pretrained weight file path, otherwise just assign a empty string.

ex:
    pretrained weight needed: load_weights = "./path_to_your_pretrained_weight_file"
        the pretrained weight of fw9395 dataset is at "./result/xxx.h5"
    pretrained weight not needed: load_weights = ""

If you want to train a new model, set variable "training = True"
If you want to test a new model, set variable "testing = True"

If you want to change the model version, change variable "model_version"
model_version = 0 -> yuting's origin PGT-Net-Block-84
model_version = 1 -> change non_binary branch: stage24*binary_output, stage60+BFM
model_version = 2 -> change binary branch: stage48+BFM
model_version = 3 -> change both branch: stage48+BFM, stage24*binary_output, stage60+BFM
model_version = 4 -> only change to add at the end of each branch: stage48+BFM, stage60+BFM
########## step 5 - run the code #########
python main.py

The testing output images and the statistic of images will be saved to the path you set in step 2.

The statistic csv file contains the MSE, SSIM, PSNR between the output testing images & their ground truth.