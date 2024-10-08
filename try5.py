#!C:\Users\alejo\Desktop\CODE\3.11\python.exe
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, UpSampling2D, GlobalAveragePooling2D, Reshape, Multiply
from tensorflow.keras.layers import AveragePooling2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50


""" Atrous Spatial Pyramid Pooling """
def ASPP(inputs):
    shape = inputs.shape

    y_pool = AveragePooling2D(pool_size=(shape[1], shape[2]), name='average_pooling')(inputs)
    y_pool = Conv2D(filters=256, kernel_size=1, padding='same', use_bias=False)(y_pool)
    y_pool = BatchNormalization(name=f'bn_1')(y_pool)
    y_pool = Activation('relu', name=f'relu_1')(y_pool)
    y_pool = UpSampling2D((shape[1], shape[2]), interpolation="bilinear")(y_pool)

    y_1 = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same', use_bias=False)(inputs)
    y_1 = BatchNormalization()(y_1)
    y_1 = Activation('relu')(y_1)

    y_6 = Conv2D(filters=256, kernel_size=3, dilation_rate=6, padding='same', use_bias=False)(inputs)
    y_6 = BatchNormalization()(y_6)
    y_6 = Activation('relu')(y_6)

    y_12 = Conv2D(filters=256, kernel_size=3, dilation_rate=12, padding='same', use_bias=False)(inputs)
    y_12 = BatchNormalization()(y_12)
    y_12 = Activation('relu')(y_12)

    y_18 = Conv2D(filters=256, kernel_size=3, dilation_rate=18, padding='same', use_bias=False)(inputs)
    y_18 = BatchNormalization()(y_18)
    y_18 = Activation('relu')(y_18)

    y = Concatenate()([y_pool, y_1, y_6, y_12, y_18])

    y = Conv2D(filters=256, kernel_size=1, dilation_rate=1, padding='same', use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    return y

def ESG(inputs, filters=256):
    # Paso 1: Global Average Pooling para capturar información global
    gap = GlobalAveragePooling2D()(inputs)
    # Paso 2: Redimensionar para la compatibilidad con la entrada
    gap = Reshape((1, 1, gap.shape[-1]))(gap)
    gap = Conv2D(filters=filters, kernel_size=1, padding='same', use_bias=False)(gap)
    gap = BatchNormalization()(gap)
    gap = Activation('relu')(gap)
    gap = Conv2D(filters=filters, kernel_size=1, padding='same', use_bias=False)(gap)
    gap = BatchNormalization()(gap)
    gap = Activation('sigmoid')(gap)
    # Paso 3: Aplicar la atención global a las características de entrada
    esg_output = Multiply()([inputs, gap])
    
    return esg_output

def L2G_ESR(low_level_features, high_level_features, filters=256):
    """ Low-to-Global Enhanced Semantic Registration Module """
    # Ajustar la resolución de high_level_features a la de low_level_features
    high_level_features_up = UpSampling2D(size=(2, 2), interpolation='bilinear')(high_level_features)
    high_level_features_up = Conv2D(filters=filters, kernel_size=1, padding='same', use_bias=False)(high_level_features_up)
    high_level_features_up = BatchNormalization()(high_level_features_up)
    high_level_features_up = Activation('relu')(high_level_features_up)

    # Concatenar low_level_features y high_level_features
    concatenated = Concatenate()([low_level_features, high_level_features_up])
    
    # Aplicar convoluciones para combinar la información
    x = Conv2D(filters=filters, kernel_size=3, padding='same', use_bias=False)(concatenated)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=filters, kernel_size=3, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x

def DeepLabV3Plus(shape):
    """ Inputs """
    inputs = Input(shape)

    """ Pre-trained ResNet50 """
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)

    """ Pre-trained ResNet50 Output """
    image_features = base_model.get_layer('conv4_block6_out').output
    x_a = ASPP(image_features)
    x_a = UpSampling2D((8, 8), interpolation="bilinear")(x_a)  # Asegúrate de que las dimensiones coincidan

    """ Get low-level features """
    x_b = base_model.get_layer('conv2_block2_out').output
    x_b = Conv2D(filters=48, kernel_size=1, padding='same', use_bias=False)(x_b)
    x_b = BatchNormalization()(x_b)
    x_b = Activation('relu')(x_b)
    x_b = UpSampling2D((4, 4), interpolation="bilinear")(x_b)  # Asegúrate de que las dimensiones coincidan

    """ Apply ESG Module to high-level features """
    x_a = ESG(x_a)

    """ Apply L2G-ESR Module to combine low and high-level features """
    x_combined = L2G_ESR(x_b, x_a)  # x_b y x_a ahora tienen las mismas dimensiones espaciales

    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', use_bias=False)(x_combined)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = UpSampling2D((4, 4), interpolation="bilinear")(x)

    """ Outputs """
    x = Conv2D(9, (1, 1), name='output_layer')(x)
    x = Activation('softmax')(x)

    """ Model """
    model = Model(inputs=inputs, outputs=x)
    return model

if __name__ == "__main__":
    input_shape = (1024, 1024, 3)
    model = DeepLabV3Plus(input_shape)
    model.summary()