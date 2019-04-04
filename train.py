import numpy as np
from src.preprocess import SplitTrainVal, EqualSplitTrainVal
from src.network import *
import keras
import math

train_label_name = './dataset/train_labels.npy'
train_img_name = './dataset/train_images.npy'
test_img_name = './dataset/test_images.npy'

import argparse

def main():


    X = np.load(train_img_name)
    y = np.load(train_label_name)

    X = np.reshape(X, [-1, 28, 28, 1])

    X = (X - 0.28603) / 0.352901

    X_tr, y_tr, X_val, y_val = EqualSplitTrainVal(X, y, validation_split=0.2)

    #train, val = keras.datasets.fashion_mnist.load_data()
    #X_tr, y_tr = train
    #X_tr = np.reshape(X_tr, (-1, 28, 28, 1))
    #X_val, y_val = val
    #X_val = np.reshape(X_val, (-1, 28, 28, 1))

    datagen = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=0,
        width_shift_range=3./28,
        height_shift_range=3./28,
        horizontal_flip=True,
        vertical_flip=False,
        zoom_range=(0.9, 1.1))
    datagen.fit(X_tr, augment=True)

    model = VGG_9(input_shape=(28, 28, 1),
                  regularizer_strength=1e-4)
    model.compile(optimizer=keras.optimizers.rmsprop(lr=0.001, rho=.9, epsilon=1e-4),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    model.summary()

    lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor='val_acc', patience=5, verbose=1, factor=.1, min_lr=1e-5)
    model_checkpoint = keras.callbacks.ModelCheckpoint('./models/model.ckpt', save_best_only=True, monitor='val_acc')

    batch_size = 128
    model.fit_generator(datagen.flow(X_tr, y_tr, batch_size=batch_size),
                        epochs=300, steps_per_epoch=len(X_tr)/batch_size,
                        validation_data=(X_val, y_val),
                        callbacks=[lr_scheduler, model_checkpoint])



if __name__ == "__main__":
    main()
