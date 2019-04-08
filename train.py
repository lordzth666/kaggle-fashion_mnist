import numpy as np
from src.preprocess import SplitTrainVal, EqualSplitTrainVal
from src.network import *
import keras
import math

train_label_name = './dataset/train_labels.npy'
train_img_name = './dataset/train_images.npy'
test_img_name = './dataset/test_images.npy'

import argparse

_mean = 0.28603
_var = 0.352901

MAX_H = 28
MAX_W = 28
MASK_RATIO = 0.25

def preprocess_func(img):
    start_w = np.random.randint(0, MAX_W)
    start_h = np.random.randint(0, MAX_H)
    w = np.random.randint(0, int(MAX_W * MASK_RATIO))
    h = np.random.randint(0, int(MAX_H * MASK_RATIO))
    end_w = min(MAX_W, start_w+w)
    end_h = min(MAX_H, start_h+h)

    rnd_values = np.random.normal(loc=_mean, scale=_var, size=[end_h-start_h, end_w-start_w])

    img_aug = img.copy()
    img_aug[start_h:end_h, start_w:end_w, 0] = rnd_values
    return img_aug

def lr_schedule(epoch, lr):
    if epoch == 2:
        return lr * 10
    else:
        if epoch == 50 or epoch == 75:
            return lr * 0.1
        else:
            return lr


def main():


    X = np.load(train_img_name)
    y = np.load(train_label_name)

    X = np.reshape(X, [-1, 28, 28, 1])

    X = (X - _mean) / _var

    X_tr, y_tr, X_val, y_val = EqualSplitTrainVal(X, y, validation_split=0.1)

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
                                        preprocessing_function=preprocess_func)
    datagen.fit(X_tr, augment=True)

    model = ResNet_18(input_shape=(28, 28, 1),
                      regularizer_strength=1e-4)

    model.compile(optimizer=keras.optimizers.sgd(lr=0.01, momentum=.9, nesterov=False),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    model.summary()

    lr_scheduler = keras.callbacks.LearningRateScheduler(schedule=lr_schedule, verbose=1)
    model_checkpoint = keras.callbacks.ModelCheckpoint('./models/model.ckpt', save_best_only=True, monitor='val_acc')

    batch_size = 128
    model.fit_generator(datagen.flow(X_tr, y_tr, batch_size=batch_size),
                        epochs=300, steps_per_epoch=len(X_tr)/batch_size,
                        validation_data=(X_val, y_val),
                        callbacks=[lr_scheduler, model_checkpoint])

if __name__ == "__main__":
    main()
