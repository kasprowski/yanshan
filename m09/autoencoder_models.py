import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input
from tensorflow.keras.layers import UpSampling2D, LeakyReLU, Conv2DTranspose, Concatenate
from tensorflow.keras.initializers import RandomNormal


def flat(input_size,code_size):
    input_img = Input(shape=(input_size,))
    code = Dense(code_size, activation='relu')(input_img)
    output_img = Dense(input_size, activation='sigmoid')(code)
    model = Model(input_img, output_img)
    return model

def flat2(input_size,code_size):
    input_img = Input(shape=(input_size,))
    hidden_1 = Dense(128, activation='relu')(input_img)
    code = Dense(code_size, activation='relu')(hidden_1)
    hidden_2 = Dense(128, activation='relu')(code)
    output_img = Dense(input_size, activation='sigmoid')(hidden_2)
    model = Model(input_img, output_img)
    return model

def upsampling_model(image_shape):
    input_img = Input(shape=image_shape)
    x = Conv2D(filters = 32, kernel_size = (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D(pool_size = (2, 2), padding='same')(x)
    x = Conv2D(filters = 16, kernel_size = (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size = (2, 2), padding='same')(x) 
    x = Conv2D(filters = 8, kernel_size = (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D(pool_size = (2, 2), padding='same', name='encoded')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    output_img = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    model = Model(input_img, output_img)
    return model

def transpose_model(image_shape):
    input_img = Input(shape=image_shape)
    x = Conv2D(filters = 32, kernel_size = (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D(pool_size = (2, 2), padding='same')(x)
    x = Conv2D(filters = 16, kernel_size = (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size = (2, 2), padding='same')(x) 
    encoded = Conv2D(filters = 8, kernel_size = (3, 3), activation='relu', padding='same', name='encoded')(x)
    x = Conv2DTranspose(8, (3, 3), activation='relu', padding='same')(encoded)
    x = Conv2DTranspose(16, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    x = Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
    output_img = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    model = Model(input_img, output_img)
    return model



# define an encoder block
def encoder_block(layer_in, n_filters, batchnorm=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    g = Conv2D(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    if batchnorm:
        g = BatchNormalization()(g, training=True)
    g = LeakyReLU(alpha=0.2)(g)
    return g

# define a decoder block
def decoder_block(layer_in, skip_in, n_filters, dropout=True):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    g = Conv2DTranspose(n_filters, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(layer_in)
    g = BatchNormalization()(g, training=True)
    # conditionally add dropout
    if dropout:
        g = Dropout(0.5)(g, training=True)
    # merge with skip connection
    g = Concatenate()([g, skip_in])
    # relu activation
    g = Activation('relu')(g)
    return g

# define the u-netowork
def unet_model(image_shape=(256,256,3)):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=image_shape)
    # encoder model
    e1 = encoder_block(in_image, 64, batchnorm=False)
    e2 = encoder_block(e1, 128)
    e3 = encoder_block(e2, 256)
    e4 = encoder_block(e3, 512)
    e5 = encoder_block(e4, 512)
    e6 = encoder_block(e5, 512)
    e7 = encoder_block(e6, 512)
    # bottleneck, no batch norm and relu
    b = Conv2D(512, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(e7)
    b = Activation('relu')(b)
    # decoder model
    d1 = decoder_block(b, e7, 512)
    d2 = decoder_block(d1, e6, 512)
    d3 = decoder_block(d2, e5, 512)
    d4 = decoder_block(d3, e4, 512, dropout=False)
    d5 = decoder_block(d4, e3, 256, dropout=False)
    d6 = decoder_block(d5, e2, 128, dropout=False)
    d7 = decoder_block(d6, e1, 64, dropout=False)
    # output
    g = Conv2DTranspose(3, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d7)
    out_image = Activation('tanh')(g)
    # define model
    model = Model(in_image, out_image)
    return model

