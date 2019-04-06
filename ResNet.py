"""
    18-layer Resnet(Mini)
"""

import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import glob
import numpy as np
import os
import random
import keras.callbacks
import time

from keras import Sequential, Input
from keras.layers import Conv2D, Dense, MaxPooling2D, Flatten, AveragePooling2D, \
                         BatchNormalization, Activation, Dropout
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.optimizers import adam
from keras.callbacks import TensorBoard



# Resnet block: batch_normal -> convolution -> activation
def unit(unit_input, filters, pool=False):
    raw_input = unit_input
    # input_dim equals residual_dim
    if pool:
        unit_input = MaxPooling2D(pool_size=(2, 2))(unit_input)
        # 1*1 kernel convolution
        raw_input = Conv2D(filters=filters, kernel_size=[1, 1],
                           strides=[2, 2], padding='same')(raw_input)

    unit_output = BatchNormalization()(unit_input)
    unit_output = Conv2D(filters=filters, kernel_size=[3, 3],
                         strides=[1, 1], padding='same',
                         kernel_initializer='truncated_normal',
                         # kernel_regularizer='l2'
                         )(unit_output)
    unit_output = Activation('relu')(unit_output)

    unit_output = BatchNormalization()(unit_output)
    unit_output = Conv2D(filters=filters, kernel_size=[3, 3],
                         strides=[1, 1], padding='same',
                         kernel_initializer='truncated_normal',
                         # kernel_regularizer='l2'
                         )(unit_output)

    unit_output = Activation('relu')(unit_output)

    return keras.layers.add([unit_output, raw_input])


def block(block_input, filters, num):
    for i in range(num):
        # if first, pool = true
        if i == 0:
            block_output = unit(block_input, filters=filters, pool=True)
            # print('pool:', filters)
        else:
            block_output = unit(block_output, filters=filters)
        # print('conv:', filters)
        # print(block_output.shape)
    return block_output


# x_input = np.ones((1, 4, 4, 3), dtype='float32')
# block(tf.convert_to_tensor(x_input), 64, 2)

# create model
def ResNet18(input_shape):
    # conv_1
    spec_input = Input(input_shape)
    # higher configuration for more data
    # net = Conv2D(filters=64, kernel_size=[3, 3], strides=[1, 1], padding='same')(spec_input)
    # # conv_2
    # net = block(net, 64, 2)
    # # conv_3
    # net = block(net, 128, 2)
    # # conv_4
    # net = block(net, 256, 2)
    # # conv_5
    # net = block(net, 512, 2)
    net = Conv2D(filters=2, kernel_size=[3, 3], strides=[1, 1], padding='same',
                 # kernel_initializer='random_uniform', bias_initializer='zeros',
                 kernel_regularizer='l2')(spec_input)
    # conv_2
    net = block(net, 2, 2)
    # conv_3
    net = block(net, 2, 2)
    # conv_4
    net = block(net, 8, 2)
    # conv_5
    net = block(net, 16, 2)
    # full_connect 512
    net = AveragePooling2D(pool_size=(8, 8))(net)
    # dropout
    net = Flatten()(net)
    net = Dropout(0.5)(net)
    net = Dense(units=8, activation='softmax')(net)
    model = Model(inputs=spec_input, outputs=net)
    return model


# model = ResNet18(np.ones((128, 128, 3), dtype='float32'), (128,128,3))

# read data
data = []
label = []
train_data_dir = '/home/wangz/Desktop/音乐推荐系统/train_data'
spectrum_list = glob.glob(os.path.join(train_data_dir, '*/*.jpg'))

for spectrum_path in spectrum_list:
    spectrum = plt.imread(spectrum_path)
    data.append(spectrum)
    label.append(spectrum_path.split('/')[-2])


# shuffle the dataset
def shuffle_list(data, label):
    list = []
    shuffled_data = []
    shuffled_labels = []
    for d, l in zip(data, label):
        list.append((d, l))
    random.seed(16)
    random.shuffle(list)
    for dl_tuple in list:
        shuffled_data.append(dl_tuple[0])
        shuffled_labels.append(dl_tuple[1])
    return shuffled_data, shuffled_labels


data, label = shuffle_list(data, label)
# print(label)

# 编码格式:
# ChaCha        : 1
# Jive          : 2
# Quickstep     : 3
# Rumba         : 4
# Samba         : 5
# SlowWaltz     : 6
# Tango         : 7
# VienneseWaltz : 8
encoder = LabelEncoder()
label = encoder.fit_transform(label)
label = np_utils.to_categorical(label, 8)
#
# print(label[:30])
# change list to array
data = np.array(data)

# split train & test dataset
train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.045)
# pre-process: rescale, not dataAnugmentation, not ZCA

# normalize the data
train_data = train_data.astype('float32') / 255
test_data = test_data.astype('float32') / 255

# subtract the mean image from both train and test set
train_data = train_data - train_data.mean()
test_data = test_data - test_data.mean()

# divide by the standard deviation
train_data = train_data / train_data.std(axis=0)
test_data = test_data / test_data.std(axis=0)

# augment the dataset
datagen = ImageDataGenerator(width_shift_range=5. / 32,
                             height_shift_range=5. / 32,
                             horizontal_flip=True)

datagen.fit(train_data)

spec_shape = (128, 128, 3)
model = ResNet18(spec_shape)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=adam(lr=0.005), metrics=['accuracy'])


# use tensorboard to record the procession
s_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
logs_path = '/home/wangz/Desktop/音乐推荐系统/logs/log_%s' % s_time

tensorboard = TensorBoard(log_dir=logs_path)

# fit the model on train dataset
model.fit_generator(datagen.flow(train_data, train_label, batch_size=96),
                    epochs=400, steps_per_epoch=20, shuffle=True, workers=4,
                    validation_data=[test_data, test_label], verbose=2, callbacks=[tensorboard])

# evaluate the accuracy
loss, accuracy = model.evaluate(test_data, test_label)
print(accuracy)

# save the model and weights
model.save('model.h5')

