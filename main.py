from operator import is_
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, History, CSVLogger
import loaddata
import total_loss

from pgt_block_84 import pgt_block_84
import h5py

import numpy as np
import cv2
from PIL import Image
import os
from tensorflow.python.client import device_lib

import tensorflow as tf
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from LossLearningRateScheduler import LossLearningRateScheduler
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
# sess = tf.compat.v1.Session(config=config)
print("tf.config.experimental.list_physical_devices('GPU') = ",
      tf.config.experimental.list_physical_devices('GPU'))
print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))


############################ self-defined training variables ############################
# training & validation dataset path
dataset_root = '/local/disk1/dataset/training_validation/nasic9395_1008_v1.npz'
save_name = '1101_epoch_ft'																				# filename of model
result_path = './result/nasic9395_1018_origin/'																				# path of model

############################ self-defined testing variables ############################
# binary testset ground truth
b_normal_img_path = '/local/disk1/dataset/testing/FW9395_dataset/testset/binary/y_test'										                                # binary testset ground truth
# non-binary testset input
nb_test_img_path = '/local/disk1/dataset/recognition_fingerprints/FW9395/collection1/identify'										                            # non-binary testset input
# non-binary testset ground truth
nb_normal_img_path = '/local/disk1/dataset/dataset/testing/FW9395_dataset/testset/non_binary/y_test'									                                # non-binary testset ground truth

# path of binary testing output
b_test_result_path = './testing/collection1_origin/binary'
# path of non-binary testing output
nb_test_result_path = '/local/disk1/dataset/recognition_fingerprints/FW9395/collection1/nasic9395_1008_origin'

# path of output statistic file
csv_path = "./testing/collection1_origin/nasic9395_1018_origin_176x36.csv"
# path of output learning rate statistic file (debug)
lr_csv_path = "./testing/collection1_origin/nasic9395_1018_origin_176x36_lr.csv"

############################ self-defined other variables ############################
#lr = 0.001
lr = 0.001
epochs = 100
batch_size = 15
load_weights = ''  # pretrain weight path
shape_testing = False
model_version = 0
training = True
testing = True
data_augmentation = False
binary_only = False

print('============================Setting================================')
print('dataset_root = ', dataset_root)
print('save_name = ', save_name)
print('result_path = ', result_path)
print('epochs = ', epochs)
print('batch_size = ', batch_size)
print('load_weights = ', load_weights)
print('shape_testing = ', shape_testing)
print('training = ', training)
print('testing = ', testing)
if model_version == 0:
    print('version: thesis')
if model_version == 1:
    print('version: right-hand change')
if model_version == 2:
    print('version: left-hand change')
if model_version == 3:
    print('version: all change')
if model_version == 4:
    print('version: two add')
print('===================================================================')


def data_aug(data_augmentation):
    dataset_xtrain = dataset["x_train"].astype("float16") / 255.0
    dataset_ytrain = dataset["y_train"].astype("float16") / 255.0
    dataset_xval = dataset["x_val"].astype("float16") / 255.0
    dataset_yval = dataset["y_val"].astype("float16") / 255.0

    train_cnt = dataset_xtrain.shape[0]
    img_size_h = dataset_xtrain.shape[1]
    img_size_w = dataset_xtrain.shape[2]

    for i in range(dataset_xtrain.shape[0]):
        if(i % 3 == 0):
            train_cnt = train_cnt + 1

    dataset_xtrain_aug = dataset_xtrain
    dataset_ytrain_aug = dataset_ytrain

    if(data_augmentation):
        dataset_xtrain_aug = np.empty(
            [train_cnt, img_size_h, img_size_w, 2], dtype=np.float16)
        dataset_ytrain_aug = np.empty(
            [train_cnt, img_size_h, img_size_w, 2], dtype=np.float16)

        print("dataset_xtrain.shape = ", dataset_xtrain.shape)

        for i in range(dataset_xtrain.shape[0]):
            dataset_xtrain_aug[i] = dataset_xtrain[i]
            dataset_ytrain_aug[i] = dataset_ytrain[i]

        train_cnt = dataset_xtrain.shape[0] - 1
        for i in range(dataset_xtrain.shape[0]):
            if(i % 3 == 0):
                train_cnt = train_cnt + 1
                flipcode = np.random.randint(0, high=2)

                img_aug_x = np.flip(dataset_xtrain[i], flipcode)
                img_aug_y = np.flip(dataset_ytrain[i], flipcode)
                img_aug_x = np.expand_dims(img_aug_x, axis=0)
                img_aug_y = np.expand_dims(img_aug_y, axis=0)
                print("img_aug_x.shape = ", img_aug_x.shape)
                dataset_xtrain_aug[train_cnt] = img_aug_x
                dataset_ytrain_aug[train_cnt] = img_aug_y

                if(i == 0):
                    img_result = np.squeeze(img_aug_x)[:, :, 0]
                    img_result = (np.clip(img_result*255, 0, 255)
                                  ).astype(np.uint8)
                    cv2.imwrite('./test_result_flip.bmp', img_result)

                    img_result = np.squeeze(dataset_xtrain[i])[:, :, 0]
                    img_result = (np.clip(img_result*255, 0, 255)
                                  ).astype(np.uint8)
                    cv2.imwrite('./test_result.bmp', img_result)

                    img_result = np.squeeze(img_aug_x)[:, :, 1]
                    img_result = (np.clip(img_result*255, 0, 255)
                                  ).astype(np.uint8)
                    cv2.imwrite('./test_result_flip_b.bmp', img_result)

                    img_result = np.squeeze(dataset_xtrain[i])[:, :, 1]
                    img_result = (np.clip(img_result*255, 0, 255)
                                  ).astype(np.uint8)
                    cv2.imwrite('./test_result_b.bmp', img_result)

                    img_result = np.squeeze(img_aug_y)[:, :, 0]
                    img_result = (np.clip(img_result*255, 0, 255)
                                  ).astype(np.uint8)
                    cv2.imwrite('./test_result_flip_y.bmp', img_result)

                    img_result = np.squeeze(dataset_ytrain[i])[:, :, 0]
                    img_result = (np.clip(img_result*255, 0, 255)
                                  ).astype(np.uint8)
                    cv2.imwrite('./test_result_y.bmp', img_result)

                    img_result = np.squeeze(img_aug_y)[:, :, 1]
                    img_result = (np.clip(img_result*255, 0, 255)
                                  ).astype(np.uint8)
                    cv2.imwrite('./test_result_flip_b_y.bmp', img_result)

                    img_result = np.squeeze(dataset_ytrain[i])[:, :, 1]
                    img_result = (np.clip(img_result*255, 0, 255)
                                  ).astype(np.uint8)
                    cv2.imwrite('./test_result_b_y.bmp', img_result)

    return dataset_xtrain_aug[:, :, :, 0], dataset_ytrain_aug, dataset_xval[:, :, :, 0], dataset_yval


dataset = loaddata.load_data(dataset_root)
dataset_xtrain, dataset_ytrain, dataset_xval, dataset_yval = data_aug(
    data_augmentation)
print(dataset_xtrain.shape)
print(dataset_ytrain.shape)
print(dataset_xval.shape)
print(dataset_yval.shape)


def scheduler(epoch, lr):
    if(epoch == 0):
        with open(lr_csv_path, "w") as f:
            f.write("epoch,lr\n")
            if epoch < 40:
                print("epoch = %d, lr = %f" % (epoch, lr))
                f.write("epoch = %d, lr = %f" % (epoch, lr))
                return lr
            else:
                print("epoch = %d, lr = %f" % (epoch, lr * 0.9))
                f.write("epoch = %d, lr = %f" % (epoch, lr * 0.9))
                return lr * 0.9
    else:
        with open(lr_csv_path, "a") as f:
            if epoch < 40:
                print("epoch = %d, lr = %f" % (epoch, lr))
                f.write("epoch = %d, lr = %f" % (epoch, lr))
                return lr
            else:
                print("epoch = %d, lr = %f" % (epoch, lr * 0.9))
                f.write("epoch = %d, lr = %f" % (epoch, lr * 0.9))
                return lr * 0.9


def train_binary_only(model):
    for l in model.layers:
        print(l.name.split("_")[len(l.name.split("_")) - 1])
        if(l.name.split("_")[len(l.name.split("_")) - 1] == "nb"):
            l.trainable = False


def train_all(model):
    for l in model.layers:
        l.trainable = True


def main():                    
    model = pgt_block_84(model_version=model_version,
                         is_shape_testing=shape_testing)    

    if shape_testing:
        model.summary()

    if not shape_testing:
        if load_weights != '':
            # model.load_weights(load_weights)
            pretrain_model = pgt_block_84(model_version=0,
                         is_shape_testing=shape_testing)
            pretrain_model.load_weights(filepath=load_weights,by_name=True, skip_mismatch=True)                     
            
            for layer in model.layers:
                for pretrain_layer in pretrain_model.layers:
                    if layer.name == pretrain_layer.name:
                        origin_weights = layer.get_weights()
                        pretrain_weights = pretrain_layer.get_weights()
                        weight_index = 0
                        is_comaptible = True
                        for o_weight in origin_weights:
                            p_weight = pretrain_weights[weight_index]
                            if o_weight.shape != p_weight.shape:
                                is_comaptible=False
                            weight_index+=1
                        if is_comaptible:
                            layer.set_weights(pretrain_layer.get_weights())

        if(binary_only):
            train_binary_only(model)
        else:
            train_all(model)

        if training:
            train(model=model, dataset=dataset,
                  epochs=epochs, batch_size=batch_size)
        if testing:
            # test(model=model, dataset=dataset)
            test_recognition(model=model, dataset=dataset)
            # test_single_output_channel(model=model, dataset=dataset)


def train(model=None, dataset=None, epochs=100, batch_size=10):
    LossLearningRateScheduler_ = LossLearningRateScheduler(
        model, lr, lr_csv_path, decay_threshold=0.005, decay_rate=0.95, loss_type='loss')
    callbacks = [ModelCheckpoint(filepath=result_path+save_name+'_{epoch}.h5', save_weights_only=True, verbose=1, save_best_only=True), History(
    ), CSVLogger(result_path+save_name+"_acc.csv", append=load_weights != ''), LossLearningRateScheduler_]

    print('============================')
    print('start training...')

    adam = keras.optimizers.Adam(
        learning_rate=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-7, amsgrad=False)
    model.compile(optimizer=adam, loss=total_loss.total_loss_mse_lap_ssim)

    hist_model = model.fit(dataset_xtrain,
                           dataset_ytrain,
                           validation_data=(dataset_xval, dataset_yval),
                           batch_size=batch_size, callbacks=callbacks,
                           shuffle=True, epochs=epochs, verbose=1)


def test_single_img(model=None, dataset=None):
    print('============================')
    print('start testing...')

    img = cv2.imread('./img/ft_01_0051_5_1.bmp', cv2.IMREAD_GRAYSCALE)
    img = img.reshape((img.shape[0], img.shape[1], 1))

    figerprint_enhanced_test = model.predict([img])

    img_result = np.squeeze(figerprint_enhanced_test)
    img_result = (np.clip(img_result, 0, 255)).astype(np.uint8)

    cv2.imwrite('./test_result.bmp', img_result)


def test(model=None, dataset=None):
    print('============================')
    print('start testing...')

    counter = 0.0
    nb_total_MSE_error = 0.0
    nb_total_SSIM_error = 0.0
    nb_total_PSNR_error = 0.0
    b_total_MSE_error = 0.0
    b_total_SSIM_error = 0.0
    b_total_PSNR_error = 0.0
    with open(csv_path, "w") as f:
        f.write("filename,nb_MSE,nb_SSIM,nb_PSNR,b_MSE,b_SSIM,b_PSNR\n")

        for img_file in os.listdir(nb_test_img_path):
            print(img_file)
            nb_img = cv2.imread(nb_test_img_path + '/' +
                                img_file, cv2.IMREAD_GRAYSCALE) / 255.0
            nb_img = nb_img.reshape((1, nb_img.shape[0], nb_img.shape[1], 1))
            append_img = nb_img

            nb_img_normal = cv2.imread(
                nb_normal_img_path + "/" + img_file, cv2.IMREAD_GRAYSCALE) / 255.0
            b_img_normal = cv2.imread(
                b_normal_img_path + "/" + img_file, cv2.IMREAD_GRAYSCALE) / 255.0

            figerprint_enhanced_test = model.predict([append_img])
            all_img_result = np.squeeze(
                figerprint_enhanced_test).astype(np.float32)

            print("all_img_result.shape = ", all_img_result.shape)
            print("len(all_img_result.shape) = ", len(all_img_result.shape))
            if(len(all_img_result.shape) == 2):
                print(img_file, " error")
                continue
            nb_img_result = all_img_result[0:all_img_result.shape[0],
                                           0:all_img_result.shape[1], 0]
            print("nb_img_result.shape = ", nb_img_result.shape)
            b_img_result = all_img_result[0:all_img_result.shape[0],
                                          0:all_img_result.shape[1], 1]

            nb_mse_error = mse(nb_img_result, nb_img_normal)
            nb_ssim_error = ssim(nb_img_result, nb_img_normal)
            nb_img_result = (
                np.clip(nb_img_result*255, 0, 255)).astype(np.uint8)
            nb_img_normal = (
                np.clip(nb_img_normal*255, 0, 255)).astype(np.uint8)
            nb_psnr_error = psnr(nb_img_result, nb_img_normal)

            b_mse_error = mse(b_img_result, b_img_normal)
            b_ssim_error = ssim(b_img_result, b_img_normal)
            b_img_result = (np.clip(b_img_result*255, 0, 255)).astype(np.uint8)
            b_img_normal = (np.clip(b_img_normal*255, 0, 255)).astype(np.uint8)
            b_psnr_error = psnr(b_img_result, b_img_normal)

            cv2.imwrite(nb_test_result_path + "/" + img_file, nb_img_result)
            cv2.imwrite(b_test_result_path + "/" + img_file, b_img_result)

            counter += 1
            nb_total_MSE_error += nb_mse_error
            nb_total_SSIM_error += nb_ssim_error
            nb_total_PSNR_error += nb_psnr_error

            b_total_MSE_error += b_mse_error
            b_total_SSIM_error += b_ssim_error
            b_total_PSNR_error += b_psnr_error

            f.write(img_file)
            f.write(",")
            f.write(str(nb_mse_error))
            f.write(",")
            f.write(str(nb_ssim_error))
            f.write(",")
            f.write(str(nb_psnr_error))
            f.write(",")
            f.write(str(b_mse_error))
            f.write(",")
            f.write(str(b_ssim_error))
            f.write(",")
            f.write(str(b_psnr_error))

            f.write("\n")

    print("average nb_MSE error over %d images = %f" %
          (counter, nb_total_MSE_error/counter))
    print("average nb_SSIM error over %d images = %f" %
          (counter, nb_total_SSIM_error/counter))
    print("average nb_PSNR error over %d images = %f" %
          (counter, nb_total_PSNR_error/counter))
    print("average b_MSE error over %d images = %f" %
          (counter, b_total_MSE_error/counter))
    print("average b_SSIM error over %d images = %f" %
          (counter, b_total_SSIM_error/counter))
    print("average b_PSNR error over %d images = %f" %
          (counter, b_total_PSNR_error/counter))


def test_recognition(model=None, dataset=None):
    print('============================')
    print('start testing recognition...')

    for img_file in os.listdir(nb_test_img_path):
        print(img_file)
        if(img_file.find(".bmp") != -1):
            nb_img = cv2.imread(nb_test_img_path + '/' +
                                img_file, cv2.IMREAD_GRAYSCALE) / 255.0
            nb_img = nb_img.reshape((1, nb_img.shape[0], nb_img.shape[1], 1))
            append_img = nb_img

            figerprint_enhanced_test = model.predict([append_img])
            all_img_result = np.squeeze(
                figerprint_enhanced_test).astype(np.float32)

            print("all_img_result.shape = ", all_img_result.shape)
            print("len(all_img_result.shape) = ", len(all_img_result.shape))
            if(len(all_img_result.shape) == 2):
                print(img_file, " error")
                continue
            nb_img_result = all_img_result[0:all_img_result.shape[0],
                                           0:all_img_result.shape[1], 0]
            print("nb_img_result.shape = ", nb_img_result.shape)

            nb_img_result = (
                np.clip(nb_img_result*255, 0, 255)).astype(np.uint8)

            cv2.imwrite(nb_test_result_path + "/" + img_file, nb_img_result)


def test_single_output_channel(model=None, dataset=None):
    print('============================')
    print('start testing...')

    counter = 0.0
    nb_total_MSE_error = 0.0
    nb_total_SSIM_error = 0.0
    nb_total_PSNR_error = 0.0
    b_total_MSE_error = 0.0
    b_total_SSIM_error = 0.0
    b_total_PSNR_error = 0.0
    with open(csv_path, "w") as f:
        f.write("filename,nb_MSE,nb_SSIM,nb_PSNR,b_MSE,b_SSIM,b_PSNR\n")

        for img_file in os.listdir(nb_test_img_path):
            print(img_file)
            nb_img = cv2.imread(nb_test_img_path + '/' +
                                img_file, cv2.IMREAD_GRAYSCALE) / 255.0
            nb_img = nb_img.reshape((1, nb_img.shape[0], nb_img.shape[1], 1))

            append_img = np.append(nb_img, nb_img, axis=-1)

            nb_img_normal = cv2.imread(
                nb_normal_img_path + "/" + img_file, cv2.IMREAD_GRAYSCALE) / 255.0
            b_img_normal = cv2.imread(
                b_normal_img_path + "/" + img_file, cv2.IMREAD_GRAYSCALE) / 255.0

            figerprint_enhanced_test = model.predict([append_img])
            all_img_result = np.squeeze(
                figerprint_enhanced_test).astype(np.float32)

            print("all_img_result.shape = ", all_img_result.shape)
            print("len(all_img_result.shape) = ", len(all_img_result.shape))

            nb_img_result = all_img_result
            print("nb_img_result.shape = ", nb_img_result.shape)

            nb_mse_error = mse(nb_img_result, nb_img_normal)
            nb_ssim_error = ssim(nb_img_result, nb_img_normal)
            nb_img_result = (
                np.clip(nb_img_result*255, 0, 255)).astype(np.uint8)
            nb_img_normal = (
                np.clip(nb_img_normal*255, 0, 255)).astype(np.uint8)
            nb_psnr_error = psnr(nb_img_result, nb_img_normal)

            cv2.imwrite(nb_test_result_path + "/mse" + str(nb_mse_error) + '_' + "ssim" + str(
                nb_ssim_error) + '_' + "psnr" + str(nb_psnr_error) + img_file, nb_img_result)

            counter += 1
            nb_total_MSE_error += nb_mse_error
            nb_total_SSIM_error += nb_ssim_error
            nb_total_PSNR_error += nb_psnr_error

            f.write(img_file)
            f.write(",")
            f.write(str(nb_mse_error))
            f.write(",")
            f.write(str(nb_ssim_error))
            f.write(",")
            f.write(str(nb_psnr_error))

            f.write("\n")

    print("average nb_MSE error over %d images = %f" %
          (counter, nb_total_MSE_error/counter))
    print("average nb_SSIM error over %d images = %f" %
          (counter, nb_total_SSIM_error/counter))
    print("average nb_PSNR error over %d images = %f" %
          (counter, nb_total_PSNR_error/counter))
    print("average b_MSE error over %d images = %f" %
          (counter, b_total_MSE_error/counter))
    print("average b_SSIM error over %d images = %f" %
          (counter, b_total_SSIM_error/counter))
    print("average b_PSNR error over %d images = %f" %
          (counter, b_total_PSNR_error/counter))


def mse(imageA, imageB):

    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)

    err /= float(imageA.shape[0] * imageA.shape[1])

    return err


if __name__ == "__main__":

    main()

    print('*****************************')
    print('finish!')
