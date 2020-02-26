from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Cropping2D, Conv2DTranspose
import keras.layers
from keras.models import Model, Sequential
from keras import backend as K
from load_data import load_5layerdatagray
from make_partition import make_partition
import matplotlib.pyplot as plt
import numpy as np 
from keras.callbacks import TensorBoard
from keras import regularizers
from keras.models import load_model
import tensorflowjs as tfjs
from DataGenerator import DataGenerator
from load_data import load_5layerdatagray
import tensorflow as tf
from keras.utils import plot_model


num_epochs = 1
np.random.seed(1) # for reproducibility
unet = Sequential()

input_img = Input(shape=(572, 572, 1))
e570_64 = Conv2D(64, (3, 3), activation='relu', padding='valid')(input_img) # 570*570*64
e568_64 = Conv2D(64, (3, 3), activation='relu', padding='valid')(e570_64) # 568*568*64

e284_64 = MaxPooling2D((2,2), padding='same')(e568_64) # 284*284*64

e282_128= Conv2D(128, (3, 3), activation='relu', padding='valid')(e284_64) # 282*282*128
e280_128= Conv2D(128, (3, 3), activation='relu', padding='valid')(e282_128) # 280*280*128

e140_128 = MaxPooling2D((2,2), padding='same')(e280_128) # 140*140*128

e138_256 = Conv2D(256, (3, 3), activation='relu', padding='valid')(e140_128) # 138*138*256
e136_256 = Conv2D(256, (3, 3), activation='relu', padding='valid')(e138_256) # 136*136*256

e68_256 = MaxPooling2D((2,2), padding='same')(e136_256) # 16*16*76

e66_512 = Conv2D(512, (3, 3), activation='relu', padding='valid')(e68_256) # 66*66*512
e64_512 = Conv2D(512, (3, 3), activation='relu', padding='valid')(e68_256) # 64*64*512

e32_512 = MaxPooling2D((2,2), padding='same')(e64_512) # 32*32*1024

e30_1024 = Conv2D(1024, (3, 3), activation='relu', padding='valid')(e32_512) # 30*30*1024
e28_1024 = Conv2D(1024, (3, 3), activation='relu', padding='valid')(e30_1024) # 30*30*1024

# NOW WE START UPCONVOLVING :D
d56_512 = Conv2DTranspose(512, (2, 2), padding='valid')(e28_1024) # 56*56*512
add_56 = Cropping2D(cropping=((4, 4), (4, 4)), data_format=None)(e64_512) # crop 64*64*512 to 56*56*512
d56_1024 = concatenate([add_56, d56_512]) # add layers together to a 56*56*1024 chunk
d54_512 = Conv2D(512, (3, 3), activation='relu', padding='valid')(d56_1024) # 54*54*512
d52_512 = Conv2D(512, (3, 3), activation='relu', padding='valid')(d54_512) # 52*52*512

d104_256 = Conv2DTranspose(256, (2, 2), padding='valid')(d52_512) # upconvolve to 104*104*256
add_104 = Cropping2D(cropping=((16, 16), (16, 16)), data_format=None)(e136_256) # crop 136*136*256 to 104*104*256
d104_512 = concatenate([add_104, d104_256]) # add layers together to an 104*104*512 chunk
d102_256 = Conv2D(256, (3, 3), activation='relu', padding='valid')(d104_512) # 102*102*256
d100_256 = Conv2D(256, (3, 3), activation='relu', padding='valid')(d102_256) # 100*100**256

d200_128 = Conv2DTranspose(128, (2, 2), padding='valid')(d100_256) # upconvolve to 200*200*128
add_200 = Cropping2D(cropping=((40, 40), (40, 40)), data_format=None)(e280_128) # crop 280*280*128 to 200*200*128 chunk
d200_256 = concatenate([add_200, d200_128]) # add together to 200*200*256 chunk
d198_128 = Conv2D(128, (3, 3), activation='relu', padding='valid')(d200_128) # 198*198*128
d196_128 = Conv2D(128, (3, 3), activation='relu', padding='valid')(d198_128) # 196*196*128

d392_64 = Conv2DTranspose(64, (2, 2), padding='valid')(d196_128) # upconvolve to 392*392*64
add_392 = Cropping2D(cropping=((88, 88), (88, 88)), data_format=None)(e568_64) # crop 568*568*64 to 392*392*64
d392_128 = concatenate([add_392, d392_64]) # add together to 392*392*128 chunk
d390_64 = Conv2D(64, (3, 3), activation='relu', padding='valid')(d392_128)
d388_64 = Conv2D(64, (3, 3), activation='relu', padding='valid')(d390_64)
output = Conv2D(2, (1, 1), activation=None, padding='valid')(d388_64)

unet = Model(input_img, output)
unet.compile(optimizer='adadelta', loss= lambda y_true, y_pred: keras.losses.mean_absolute_error(y_true, y_pred))
# autoencoder.compile(optimizer='adadelta', loss= 'mean_squared_error' )

partition = make_partition('/home/vera/Documents/autoencoder/data/mcdnn_128_chunks_bigdataset/', 0.9)
training_generator = DataGenerator(partition['train'])
validation_generator = DataGenerator(partition['validation'])

# print(autoencoder.summary())

# autoencoder.fit_generator(generator=training_generator,
#                     validation_data=validation_generator,
#                     epochs=10,
#                     use_multiprocessing=True,
#                     workers=2
#                     )

x_in_train, x_val_train, x_in_test, x_val_test = load_5layerdatagray('/mcdnn_128/','/validation_128grayscale/', 0.7)
x_in_train = np.reshape(x_in_train, (len(x_in_train), 128, 128, 5)) 
x_in_test = np.reshape(x_in_test, (len(x_in_test), 128, 128, 5))
x_val_train = np.reshape(x_val_train, (len(x_val_train), 128, 128, 1))  
x_val_test = np.reshape(x_val_test, (len(x_val_test), 128, 128, 1))  
decoded_imgs = unet.predict(x_in_test)

# autoencoder.save("Nmodel_temp.h5")
# tfjs.converters.save_keras_model(autoencoder, "nvidia_skipsandgrayscaledatamodel_tfjs");

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(3, n, i+1)
    pic = x_val_test[i].reshape(128, 128, 1)
    plt.imshow(pic[:,:,0], cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display input
    ax = plt.subplot(3, n, i + n + 1)
    plt.imshow(x_in_test[i,:,:,1], cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(3, n, i + n  + n + 1)
    pic = decoded_imgs[i].reshape(128, 128, 1)
    plt.imshow(pic[:,:,0], cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
#plt.savefig("shallow_nvidia_copy_skipsandgray_withbigdataset_"+str(num_epochs)+"epochs.png")
plt.show()