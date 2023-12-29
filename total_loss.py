import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.python.ops.numpy_ops import np_config
import sys
np_config.enable_numpy_behavior()

alpha = 0.2
def total_loss(y_true, y_pred): 
    

    mse = keras.losses.MeanSquaredError()(y_true, y_pred)
    
    #lapKernel = K.constant([[-1,-1,-1],[-1,8,-1],[-1,-1,-1], [-1,-1,-1],[-1,8,-1],[-1,-1,-1]], shape = [3, 3, 1, 2])
    lapKernel = K.constant([[-1, -1],[-1, -1],[-1, -1],[-1, -1],[8, 8],[-1, -1],[-1, -1],[-1, -1],[-1, -1]], shape = [3, 3, 1, 2])
    
    
    trueLap = K.conv2d(y_true.astype('float32'), lapKernel)
    predLap = K.conv2d(y_pred.astype('float32'), lapKernel)
    lap = K.sum(K.square(trueLap - predLap), axis=-1)
    
    
    loss = (1-alpha) * mse + alpha * lap
    
    return loss

def enhanced_loss(y_true, y_pred):
    mse = keras.losses.MeanAbsoluteError()((y_true==0), y_pred)
    return mse

def total_loss_mse_no_reduction(y_true, y_pred): 

    mse_channel_0 = keras.losses.MeanSquaredError()(y_true[:,:,:,0], y_pred[:,:,:,0])
    mse_channel_1 = keras.losses.MeanSquaredError()(y_true[:,:,:,1], y_pred[:,:,:,1])
    
    lapKernel = K.constant([[-1, -1],[-1, -1],[-1, -1],[-1, -1],[8, 8],[-1, -1],[-1, -1],[-1, -1],[-1, -1]], shape = [3, 3, 1, 2])
    
    
    trueLap = K.conv2d(y_true.astype('float32'), lapKernel)
    predLap = K.conv2d(y_pred.astype('float32'), lapKernel)

    lap = K.square(trueLap - predLap)
    
    loss_channel_0 = (1-alpha) * mse_channel_0 + alpha * lap[:,:,:,0]
    loss_channel_1 = (1-alpha) * mse_channel_1 + alpha * lap[:,:,:,1]
    

    return 0.7*loss_channel_0 + 0.3*loss_channel_1

def total_loss_mse_lap_ssim(y_true, y_pred): 

    mse_channel_0 = keras.losses.MeanSquaredError()(y_true[:,:,:,0], y_pred[:,:,:,0])
    mse_channel_1 = keras.losses.MeanSquaredError()(y_true[:,:,:,1], y_pred[:,:,:,1])
    
    lapKernel = K.constant([[-1, -1],[-1, -1],[-1, -1],[-1, -1],[8, 8],[-1, -1],[-1, -1],[-1, -1],[-1, -1]], shape = [3, 3, 1, 2])
    
    
    trueLap = K.conv2d(y_true.astype('float32'), lapKernel)
    predLap = K.conv2d(y_pred.astype('float32'), lapKernel)

    lap = K.square(trueLap - predLap)

    ssim_channel_0 = 1.0 - tf.math.reduce_mean(tf.image.ssim(tf.expand_dims(y_true[:,:,:,0], axis=-1), tf.expand_dims(y_pred[:,:,:,0], axis=-1), 1.0))
    ssim_channel_1 = 1.0 - tf.math.reduce_mean(tf.image.ssim(tf.expand_dims(y_true[:,:,:,1], axis=-1), tf.expand_dims(y_pred[:,:,:,1], axis=-1), 1.0))
    
    loss_channel_0 = 0.1 * mse_channel_0 + 0.2 * lap[:,:,:,0] + 0.7 * ssim_channel_0
    loss_channel_1 = 0.1 * mse_channel_1 + 0.2 * lap[:,:,:,1] + 0.7 * ssim_channel_1

    return 0.7*loss_channel_0 + 0.3*loss_channel_1
    # return loss_channel_0

def total_loss_mse_lap_ssim_binary_only(y_true, y_pred): 

    mse_channel_0 = keras.losses.MeanSquaredError()(y_true[:,:,:,0], y_pred[:,:,:,0])
    mse_channel_1 = keras.losses.MeanSquaredError()(y_true[:,:,:,1], y_pred[:,:,:,1])
    
    lapKernel = K.constant([[-1, -1],[-1, -1],[-1, -1],[-1, -1],[8, 8],[-1, -1],[-1, -1],[-1, -1],[-1, -1]], shape = [3, 3, 1, 2])
    
    
    trueLap = K.conv2d(y_true.astype('float32'), lapKernel)
    predLap = K.conv2d(y_pred.astype('float32'), lapKernel)

    lap = K.square(trueLap - predLap)

    ssim_channel_0 = 1.0 - tf.math.reduce_mean(tf.image.ssim(tf.expand_dims(y_true[:,:,:,0], axis=-1), tf.expand_dims(y_pred[:,:,:,0], axis=-1), 1.0))
    ssim_channel_1 = 1.0 - tf.math.reduce_mean(tf.image.ssim(tf.expand_dims(y_true[:,:,:,1], axis=-1), tf.expand_dims(y_pred[:,:,:,1], axis=-1), 1.0))
    
    loss_channel_0 = 0.1 * mse_channel_0 + 0.2 * lap[:,:,:,0] + 0.7 * ssim_channel_0
    loss_channel_1 = 0.1 * mse_channel_1 + 0.2 * lap[:,:,:,1] + 0.7 * ssim_channel_1

    # return 0.7*loss_channel_0 + 0.3*loss_channel_1
    return loss_channel_1

def mse_only(y_true, y_pred): 

    mse_channel_0 = keras.losses.MeanSquaredError()(y_true[:,:,:,0], y_pred[:,:,:,0])
    mse_channel_1 = keras.losses.MeanSquaredError()(y_true[:,:,:,1], y_pred[:,:,:,1])
    
    loss_channel_0 = mse_channel_0
    loss_channel_1 = mse_channel_1

    return 0.7*loss_channel_0 + 0.3*loss_channel_1
    # return loss_channel_0

def total_loss_mse_lap_ssim_3task(y_true, y_pred): 

    mse_channel_0 = keras.losses.MeanSquaredError()(y_true[:,:,:,0], y_pred[:,:,:,0])
    mse_channel_1 = keras.losses.MeanSquaredError()(y_true[:,:,:,1], y_pred[:,:,:,1])
    mse_channel_2 = keras.losses.MeanSquaredError()(y_true[:,:,:,2], y_pred[:,:,:,2])
    
    lapKernel = K.constant([[-1, -1, -1],[-1, -1, -1],[-1, -1, -1],[-1, -1, -1],[8, 8, 8],[-1, -1, -1],[-1, -1, -1],[-1, -1, -1],[-1, -1, -1]], shape = [3, 3, 1, 3])
    
    
    trueLap = K.conv2d(y_true.astype('float32'), lapKernel)
    predLap = K.conv2d(y_pred.astype('float32'), lapKernel)

    lap = K.square(trueLap - predLap)

    ssim_channel_0 = 1.0 - tf.math.reduce_mean(tf.image.ssim(tf.expand_dims(y_true[:,:,:,0], axis=-1), tf.expand_dims(y_pred[:,:,:,0], axis=-1), 1.0))
    ssim_channel_1 = 1.0 - tf.math.reduce_mean(tf.image.ssim(tf.expand_dims(y_true[:,:,:,1], axis=-1), tf.expand_dims(y_pred[:,:,:,1], axis=-1), 1.0))
    ssim_channel_2 = 1.0 - tf.math.reduce_mean(tf.image.ssim(tf.expand_dims(y_true[:,:,:,2], axis=-1), tf.expand_dims(y_pred[:,:,:,2], axis=-1), 1.0))
    
    loss_channel_0 = 0.1 * mse_channel_0 + 0.2 * lap[:,:,:,0] + 0.7 * ssim_channel_0
    loss_channel_1 = 0.1 * mse_channel_1 + 0.2 * lap[:,:,:,1] + 0.7 * ssim_channel_1
    loss_channel_2 = 0.1 * mse_channel_2 + 0.2 * lap[:,:,:,2] + 0.7 * ssim_channel_2

    return 0.4*loss_channel_0 + 0.3*loss_channel_1 + 0.3*loss_channel_2

def loss_pgt_block_48_fine_turning_task_v5(y_true, y_pred): 

    mse_channel_0 = keras.losses.MeanSquaredError()(y_true[:,:,:,0], y_pred[:,:,:,0])
    mse_channel_1 = keras.losses.MeanSquaredError()(y_true[:,:,:,1], y_pred[:,:,:,1])
    mse_channel_2 = keras.losses.MeanSquaredError()(y_true[:,:,:,2], y_pred[:,:,:,2])
    
    lapKernel = K.constant([[-1, -1, -1],[-1, -1, -1],[-1, -1, -1],[-1, -1, -1],[8, 8, 8],[-1, -1, -1],[-1, -1, -1],[-1, -1, -1],[-1, -1, -1]], shape = [3, 3, 1, 3])
    
    
    trueLap = K.conv2d(y_true.astype('float32'), lapKernel)
    predLap = K.conv2d(y_pred.astype('float32'), lapKernel)

    lap = K.square(trueLap - predLap)

    ssim_channel_0 = 1.0 - tf.math.reduce_mean(tf.image.ssim(tf.expand_dims(y_true[:,:,:,0], axis=-1), tf.expand_dims(y_pred[:,:,:,0], axis=-1), 1.0))
    ssim_channel_1 = 1.0 - tf.math.reduce_mean(tf.image.ssim(tf.expand_dims(y_true[:,:,:,1], axis=-1), tf.expand_dims(y_pred[:,:,:,1], axis=-1), 1.0))
    ssim_channel_2 = 1.0 - tf.math.reduce_mean(tf.image.ssim(tf.expand_dims(y_true[:,:,:,2], axis=-1), tf.expand_dims(y_pred[:,:,:,2], axis=-1), 1.0))
    
    loss_channel_0 = 0.7 * mse_channel_0 + 0.3 * ssim_channel_0
    loss_channel_1 = 0.1 * mse_channel_1 + 0.2 * lap[:,:,:,1] + 0.7 * ssim_channel_1
    loss_channel_2 = 0.1 * mse_channel_2 + 0.2 * lap[:,:,:,2] + 0.7 * ssim_channel_2

    return 0.4*loss_channel_0 + 0.3*loss_channel_1 + 0.3*loss_channel_2

def total_loss_mse_lap_ssim_correct(y_true, y_pred): 

    mse_channel_0 = keras.losses.MeanSquaredError()(y_true[:,:,:,0], y_pred[:,:,:,0])
    mse_channel_1 = keras.losses.MeanSquaredError()(y_true[:,:,:,1], y_pred[:,:,:,1])
    
    lapKernel = K.constant([[-1, -1],[-1, -1],[-1, -1],[-1, -1],[8, 8],[-1, -1],[-1, -1],[-1, -1],[-1, -1]], shape = [3, 3, 1, 2])
    
    
    trueLap = K.conv2d(y_true.astype('float32'), lapKernel)
    predLap = K.conv2d(y_pred.astype('float32'), lapKernel)

    lap = K.square(trueLap - predLap)

    ssim_channel_0 = 1.0 - tf.math.reduce_mean(tf.image.ssim(tf.expand_dims(y_true[:,:,:,0], axis=-1), tf.expand_dims(y_pred[:,:,:,0], axis=-1), 1.0))
    ssim_channel_1 = 1.0 - tf.math.reduce_mean(tf.image.ssim(tf.expand_dims(y_true[:,:,:,1], axis=-1), tf.expand_dims(y_pred[:,:,:,1], axis=-1), 1.0))
    
    loss_channel_0 = 0.7 * mse_channel_0 + 0.2 * lap[:,:,:,0] + 0.1 * ssim_channel_0
    loss_channel_1 = 0.7 * mse_channel_1 + 0.2 * lap[:,:,:,1] + 0.1 * ssim_channel_1

    return 0.7*loss_channel_0 + 0.3*loss_channel_1
    # return loss_channel_0

def total_loss_mae_lap_ssim(y_true, y_pred): 

    mae_channel_0 = keras.losses.MeanAbsoluteError()(y_true[:,:,:,0], y_pred[:,:,:,0])
    mae_channel_1 = keras.losses.MeanAbsoluteError()(y_true[:,:,:,1], y_pred[:,:,:,1])
    
    lapKernel = K.constant([[-1, -1],[-1, -1],[-1, -1],[-1, -1],[8, 8],[-1, -1],[-1, -1],[-1, -1],[-1, -1]], shape = [3, 3, 1, 2])
    
    
    trueLap = K.conv2d(y_true.astype('float32'), lapKernel)
    predLap = K.conv2d(y_pred.astype('float32'), lapKernel)

    lap = K.square(trueLap - predLap)

    ssim_channel_0 = 1.0 - tf.math.reduce_mean(tf.image.ssim(tf.expand_dims(y_true[:,:,:,0], axis=-1), tf.expand_dims(y_pred[:,:,:,0], axis=-1), 1.0))
    ssim_channel_1 = 1.0 - tf.math.reduce_mean(tf.image.ssim(tf.expand_dims(y_true[:,:,:,1], axis=-1), tf.expand_dims(y_pred[:,:,:,1], axis=-1), 1.0))
    
    loss_channel_0 = 0.1 * mae_channel_0 + 0.2 * lap[:,:,:,0] + 0.7 * ssim_channel_0
    loss_channel_1 = 0.1 * mae_channel_1 + 0.2 * lap[:,:,:,1] + 0.7 * ssim_channel_1

    return 0.7*loss_channel_0 + 0.3*loss_channel_1

def total_loss_mae_lap_ssim_correct(y_true, y_pred): 

    mae_channel_0 = keras.losses.MeanAbsoluteError()(y_true[:,:,:,0], y_pred[:,:,:,0])
    mae_channel_1 = keras.losses.MeanAbsoluteError()(y_true[:,:,:,1], y_pred[:,:,:,1])
    
    lapKernel = K.constant([[-1, -1],[-1, -1],[-1, -1],[-1, -1],[8, 8],[-1, -1],[-1, -1],[-1, -1],[-1, -1]], shape = [3, 3, 1, 2])
    
    
    trueLap = K.conv2d(y_true.astype('float32'), lapKernel)
    predLap = K.conv2d(y_pred.astype('float32'), lapKernel)

    lap = K.square(trueLap - predLap)

    ssim_channel_0 = 1.0 - tf.math.reduce_mean(tf.image.ssim(tf.expand_dims(y_true[:,:,:,0], axis=-1), tf.expand_dims(y_pred[:,:,:,0], axis=-1), 1.0))
    ssim_channel_1 = 1.0 - tf.math.reduce_mean(tf.image.ssim(tf.expand_dims(y_true[:,:,:,1], axis=-1), tf.expand_dims(y_pred[:,:,:,1], axis=-1), 1.0))
    
    loss_channel_0 = 0.7 * mae_channel_0 + 0.2 * lap[:,:,:,0] + 0.1 * ssim_channel_0
    loss_channel_1 = 0.7 * mae_channel_1 + 0.2 * lap[:,:,:,1] + 0.1 * ssim_channel_1

    return 0.7*loss_channel_0 + 0.3*loss_channel_1

def binary_loss(y_true, y_pred): 

    mse_channel_0 = keras.losses.MeanSquaredError()(y_true[:,:,:,0], y_pred[:,:,:,0])
    mse_channel_1 = keras.losses.MeanSquaredError()(y_true[:,:,:,1], y_pred[:,:,:,1])
    
    lapKernel = K.constant([[-1, -1],[-1, -1],[-1, -1],[-1, -1],[8, 8],[-1, -1],[-1, -1],[-1, -1],[-1, -1]], shape = [3, 3, 1, 2])
    
    
    trueLap = K.conv2d(y_true.astype('float32'), lapKernel)
    predLap = K.conv2d(y_pred.astype('float32'), lapKernel)

    lap = K.square(trueLap - predLap)

    ssim_channel_0 = 1.0 - tf.math.reduce_mean(tf.image.ssim(tf.expand_dims(y_true[:,:,:,0], axis=-1), tf.expand_dims(y_pred[:,:,:,0], axis=-1), 1.0))
    ssim_channel_1 = 1.0 - tf.math.reduce_mean(tf.image.ssim(tf.expand_dims(y_true[:,:,:,1], axis=-1), tf.expand_dims(y_pred[:,:,:,1], axis=-1), 1.0))
    
    loss_channel_0 = 0.1 * mse_channel_0 + 0.2 * lap[:,:,:,0] + 0.7 * ssim_channel_0
    loss_channel_1 = 0.1 * mse_channel_1 + 0.2 * lap[:,:,:,1] + 0.7 * ssim_channel_1

    # return 0.7*loss_channel_0 + 0.3*loss_channel_1

    return mse_channel_1


def total_loss_08mse_02lap(y_true, y_pred): 

    mse_channel_0 = keras.losses.MeanSquaredError()(y_true[:,:,:,0], y_pred[:,:,:,0])
    mse_channel_1 = keras.losses.MeanSquaredError()(y_true[:,:,:,1], y_pred[:,:,:,1])
    
    lapKernel = K.constant([[-1, -1],[-1, -1],[-1, -1],[-1, -1],[8, 8],[-1, -1],[-1, -1],[-1, -1],[-1, -1]], shape = [3, 3, 1, 2])
    
    
    trueLap = K.conv2d(y_true.astype('float32'), lapKernel)
    predLap = K.conv2d(y_pred.astype('float32'), lapKernel)

    lap = K.square(trueLap - predLap)

    
    loss_channel_0 = 0.8 * mse_channel_0 + 0.2 * lap[:,:,:,0]
    loss_channel_1 = 0.8 * mse_channel_1 + 0.2 * lap[:,:,:,1] 

    return 0.7*loss_channel_0 + 0.3*loss_channel_1

def total_loss_08mse_02lap_single_channle(y_true, y_pred): 
    
    y_true = tf.expand_dims(y_true[:,:,:,0], -1)
    mse_channel_0 = keras.losses.MeanSquaredError()(y_true, y_pred)
    
    lapKernel = K.constant([[-1, -1, -1],[-1, 8, -1],[-1, -1, -1]], shape = [3, 3, 1, 1])
    
    
    trueLap = K.conv2d(y_true.astype('float32'), lapKernel)
    predLap = K.conv2d(y_pred.astype('float32'), lapKernel)

    lap = K.square(trueLap - predLap)

    
    loss_channel_0 = 0.8 * mse_channel_0 + 0.2 * lap

    return loss_channel_0

def total_loss_mse_lap_ssim_single_channle(y_true, y_pred): 

    mse_channel_0 = keras.losses.MeanSquaredError()(y_true[:,:,:,0], y_pred[:,:,:,0])
    
    lapKernel = K.constant([[-1, -1],[-1, -1],[-1, -1],[-1, -1],[8, 8],[-1, -1],[-1, -1],[-1, -1],[-1, -1]], shape = [3, 3, 1, 2])
    
    
    trueLap = K.conv2d(y_true.astype('float32'), lapKernel)
    predLap = K.conv2d(y_pred.astype('float32'), lapKernel)

    lap = K.square(trueLap - predLap)

    ssim_channel_0 = 1.0 - tf.math.reduce_mean(tf.image.ssim(tf.expand_dims(y_true[:,:,:,0], axis=-1), tf.expand_dims(y_pred[:,:,:,0], axis=-1), 1.0))
    
    loss_channel_0 = 0.1 * mse_channel_0 + 0.2 * lap[:,:,:,0] + 0.7 * ssim_channel_0

    return loss_channel_0


def total_loss_mse_ssim(y_true, y_pred): 

    mse_channel_0 = keras.losses.MeanSquaredError()(y_true[:,:,:,0], y_pred[:,:,:,0])
    mse_channel_1 = keras.losses.MeanSquaredError()(y_true[:,:,:,1], y_pred[:,:,:,1])

    ssim_channel_0 = 1.0 - tf.math.reduce_mean(tf.image.ssim(tf.expand_dims(y_true[:,:,:,0], axis=-1), tf.expand_dims(y_pred[:,:,:,0], axis=-1), 1.0))
    ssim_channel_1 = 1.0 - tf.math.reduce_mean(tf.image.ssim(tf.expand_dims(y_true[:,:,:,1], axis=-1), tf.expand_dims(y_pred[:,:,:,1], axis=-1), 1.0))
    
    loss_channel_0 = 0.2 * mse_channel_0 + 0.8 * ssim_channel_0
    loss_channel_1 = 0.2 * mse_channel_1 + 0.8 * ssim_channel_1

    return 0.9*loss_channel_0 + 0.1*loss_channel_1

def total_loss_mae_ssim(y_true, y_pred): 

    mse_channel_0 = keras.losses.MeanAbsoluteError()(y_true[:,:,:,0], y_pred[:,:,:,0])
    mse_channel_1 = keras.losses.MeanAbsoluteError()(y_true[:,:,:,1], y_pred[:,:,:,1])

    ssim_channel_0 = 1.0 - tf.math.reduce_mean(tf.image.ssim(tf.expand_dims(y_true[:,:,:,0], axis=-1), tf.expand_dims(y_pred[:,:,:,0], axis=-1), 1.0))
    ssim_channel_1 = 1.0 - tf.math.reduce_mean(tf.image.ssim(tf.expand_dims(y_true[:,:,:,1], axis=-1), tf.expand_dims(y_pred[:,:,:,1], axis=-1), 1.0))
    
    loss_channel_0 = 0.2 * mse_channel_0 + 0.8 * ssim_channel_0
    loss_channel_1 = 0.2 * mse_channel_1 + 0.8 * ssim_channel_1

    return 0.9*loss_channel_0 + 0.1*loss_channel_1


def total_loss_08mse_02ssim(y_true, y_pred): 

    mse_channel_0 = keras.losses.MeanSquaredError()(y_true[:,:,:,0], y_pred[:,:,:,0])
    mse_channel_1 = keras.losses.MeanSquaredError()(y_true[:,:,:,1], y_pred[:,:,:,1])

    ssim_channel_0 = 1.0 - tf.math.reduce_mean(tf.image.ssim(tf.expand_dims(y_true[:,:,:,0], axis=-1), tf.expand_dims(y_pred[:,:,:,0], axis=-1), 1.0))
    ssim_channel_1 = 1.0 - tf.math.reduce_mean(tf.image.ssim(tf.expand_dims(y_true[:,:,:,1], axis=-1), tf.expand_dims(y_pred[:,:,:,1], axis=-1), 1.0))
    
    loss_channel_0 = 0.8 * mse_channel_0 + 0.2 * ssim_channel_0
    loss_channel_1 = 0.8 * mse_channel_1 + 0.2 * ssim_channel_1

    return 0.7*loss_channel_0 + 0.3*loss_channel_1

def MSE(y_true, y_pred): 
    mse = keras.losses.MeanSquaredError()(y_true, y_pred)
    return mse