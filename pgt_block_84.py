import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, History, CSVLogger
import loaddata
import total_loss


def Residual_block(input_tensor, block, scale, partial_name="mix"):

    name_base = 'Block' + str(block)

    x = tf.keras.layers.Conv2D(
        64, (3, 3), padding='same', name=name_base + '_conv1_' + partial_name)(input_tensor)

    if (partial_name == "b"):
        x = tf.keras.layers.Activation(
            'sigmoid', name=name_base + '_sigmoid_' + partial_name)(x)
    else:
        x = tf.keras.layers.Activation(
            'relu', name=name_base + '_relu_' + partial_name)(x)

    x = tf.keras.layers.Conv2D(
        64, (3, 3), padding='same', name=name_base + '_conv2_' + partial_name)(x)

    x = tf.keras.layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                               output_shape=K.int_shape(x)[1:],
                               arguments={'scale': scale}, name=name_base + '_add_' + partial_name)([input_tensor, x])
    return x


def Residual_block_branch_pt(input_tensor, block, scale, partial_name="mix"):

    name_base = 'Block' + str(block)

    x = tf.keras.layers.Conv2D(
        64, (3, 3), padding='same', name=name_base + '_conv1_' + partial_name)(input_tensor)

    if (partial_name == "b"):
        x = tf.keras.layers.Activation(
            'sigmoid', name=name_base + '_sigmoid_' + partial_name)(x)
    else:
        x = tf.keras.layers.Activation(
            'relu', name=name_base + '_relu_' + partial_name)(x)

    x = tf.keras.layers.Conv2D(
        64, (3, 3), padding='same', name=name_base + '_conv2_' + partial_name)(x)

    input_tensor = tf.keras.layers.Conv2D(
        64, (1, 1), padding='same', name=name_base + '_pointwise_conv_' + partial_name)(input_tensor)

    x = tf.keras.layers.Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
                               output_shape=K.int_shape(x)[1:],
                               arguments={'scale': scale}, name=name_base + '_add_' + partial_name)([input_tensor, x])
    return x


def pgt_block_84(output_branch=1, block_num=48, total_stages=24, model_version=0, is_shape_testing=False):
    input_tensor = tf.keras.layers.Input(shape=(None, None, 1), name="input")
    non_binary_input = tf.expand_dims(input_tensor[:, :, :, 0], -1)

    BFM = tf.keras.layers.Conv2D(
        128, 3, dilation_rate=3, padding='same', name='BFM_dilated1')(non_binary_input)
    BFM = tf.keras.layers.Conv2D(
        256, 3, dilation_rate=2, padding='same', name='BFM_dilated2')(BFM)

    ####################################################
    # process shared features
    ####################################################
    for block in range(1, block_num//2 + 1):
        scale = total_stages*0.01-0.01*block
        if(scale <= 0):
            scale = scale - 0.01

        if block == 1:
            x = Residual_block_branch_pt(BFM, block, scale)
        else:
            x = Residual_block(x, block, scale)

    ####################################################
    # process binary features
    ####################################################
    for block in range(block_num//2 + 1, block_num+1 + 12):
        scale = total_stages*0.01-0.01*block
        if(scale <= 0):
            scale = scale - 0.01

        if block == (block_num//2 + 1):
            b_x = Residual_block_branch_pt(x, block, scale, "b")
        else:
            b_x = Residual_block(b_x, block, scale, "b")

    if (model_version == 0) or (model_version == 1):
        # origin thesis: concatenate
        b_x = tf.keras.layers.Concatenate(name='Superimpose_b')([b_x, BFM])
        if is_shape_testing:
            print('model_version: ', model_version)
            print('b_x shape: ', b_x.shape)
            print('BFM shape: ', BFM.shape)
        ###
    elif (model_version == 2) or (model_version == 3) or (model_version == 4):
        # version2 testing: concatenate to add
        BFM_conv = tf.keras.layers.Conv2D(
            64, (1, 1), padding='same', name='BFM_b_pointwise_conv_')(BFM)

        if is_shape_testing:
            print('model_version: ', model_version)
            print('b_x shape: ', b_x.shape)
            print('BFM_conv shape: ', BFM_conv.shape)

        b_x = tf.keras.layers.Add(name='Add_denoise_b')([b_x, BFM_conv])
        ###

    b_x = tf.keras.layers.Conv2D(
        1, (1, 1), padding='same', name='final_conv_b')(b_x)

    if is_shape_testing:
        print('b_x shape: ', b_x.shape)
        print('x shape: ', x.shape)

    ####################################################
    # process non-binary features
    ####################################################

    if (model_version == 0) or (model_version == 2) or (model_version == 4):
        # origin thesis: concatenate
        x_bx = tf.keras.layers.Concatenate(name='Superimpose_xb')(
            [x, b_x])  # change this line to multiple
        if is_shape_testing:
            print('x_bx shape: ', x_bx.shape)
    elif (model_version == 1) or (model_version == 3):
        # version1 testing: concatenate to add
        x_bx = tf.keras.layers.Multiply(
            name='multiply_Superimpose_xb')([x, b_x])
        if is_shape_testing:
            print('x_bx shape: ', x_bx.shape)

    for block in range(block_num//2 + 1, block_num//2 + 1 + 24):
        scale = total_stages*0.01-0.01*block
        if(scale <= 0):
            scale = scale - 0.01

        if block == (block_num//2 + 1):
            nb_x = Residual_block_branch_pt(x_bx, block, scale, "nb")
        else:
            nb_x = Residual_block(nb_x, block, scale, "nb")

    if (model_version == 0) or (model_version == 2):
        # origin thesis: concatenate
        nb_x = tf.keras.layers.Concatenate(
            name='Superimpose_nb')([nb_x, BFM])  # change to add
    elif (model_version == 1) or (model_version == 3) or (model_version == 4):
        # version1 testing: concatenate to add
        BFM_conv = tf.keras.layers.Conv2D(
            64, (1, 1), padding='same', name='BFM_b_pointwise_conv_2')(BFM)
        nb_x = tf.keras.layers.Add(name='Add_Superimpose_nb')(
            [nb_x, BFM_conv])  # change to add

    nb_x = tf.keras.layers.Conv2D(
        1, (1, 1), padding='same', name='final_conv_nb')(nb_x)

    x = tf.keras.layers.Concatenate(name='final_concatenate')([nb_x, b_x])

    if output_branch == 1:
        model = tf.keras.models.Model(inputs=input_tensor, outputs=x)
    else:
        model = tf.keras.models.Model(inputs=input_tensor, outputs=[x, x])
    return model


if __name__ == "__main__":
    model = pgt_block_84()

    model.summary()
