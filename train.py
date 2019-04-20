import numpy as np
from src.preprocess import SplitTrainVal, EqualSplitTrainVal
from src.network import *
import keras
import math

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.20
set_session(tf.Session(config=config))

train_label_name = './dataset/train_labels.npy'
train_img_name = './dataset/train_images.npy'
test_img_name = './dataset/test_images.npy'

import argparse

_mean = 0.28603
_var = 0.352901

MAX_H = 28
MAX_W = 28
MASK_RATIO = 0.50


MAX_EPOCHS = 100
WARMUP_EPOCH = 1
WARMUP_DECAY = 0.75
LR_DECAY = 0.1
INITIAL_LR = 0.01
DECAY_EPOCH = [MAX_EPOCHS*0.30, MAX_EPOCHS*0.60, MAX_EPOCHS*0.90]

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
    if epoch == 0:
        return lr * WARMUP_DECAY
    elif epoch == WARMUP_EPOCH:
        return lr / WARMUP_DECAY
    else:
        if epoch in DECAY_EPOCH:
            return lr * LR_DECAY
        else:
            return lr


def main(args):


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
                                        width_shift_range=0./28,
                                        height_shift_range=0./28,
                                        horizontal_flip=False,
                                        vertical_flip=False)

    datagen.fit(X_tr, augment=True)

    # WRN-28-2
    #model = Wide_ResNet(input_shape=(28, 28, 1),
    #                    regularizer_strength=1e-4,
    #                    n=4,
    #                    k=2,
    #                    dropout=.1)

    # VGG-10
    model = VGG_10(input_shape=(28, 28, 1),
                   regularizer_strength=1e-16,
                   dropout=0.5,
                   batchnorm=True)

    model.compile(optimizer=keras.optimizers.sgd(lr=INITIAL_LR, momentum=.9, nesterov=True),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    model.summary()

    lr_scheduler = keras.callbacks.LearningRateScheduler(schedule=lr_schedule, verbose=1)
    model_checkpoint = keras.callbacks.ModelCheckpoint(args['save_dir'], save_best_only=True, monitor='val_acc')

    batch_size = 256
    model.fit_generator(datagen.flow(X_tr, y_tr, batch_size=batch_size),
                        epochs=MAX_EPOCHS, steps_per_epoch=len(X_tr)/batch_size,
                        validation_data=(X_val, y_val),
                        callbacks=[lr_scheduler, model_checkpoint])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", required=True, type=str, help="Model save dir", default=None)
    args = vars(parser.parse_args())
    main(args)
