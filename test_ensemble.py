import numpy as np
from src.preprocess import SplitTrainVal
from src.network import *
import keras

import argparse
import os

train_label_name = './dataset/train_labels.npy'
train_img_name = './dataset/train_images.npy'
test_img_name = './dataset/test_images.npy'

_mean = 0.28603
_var = 0.352901

def main(args):
    X = np.load(test_img_name)
    X = np.reshape(X, [-1, 28, 28, 1])
    X = (X - _mean) / _var

    model = VGG_10(input_shape=(28, 28, 1),
                   regularizer_strength=2e-4)
    model.compile(optimizer=keras.optimizers.sgd(lr=1e-2, momentum=.9),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    model.summary()
    model_list = os.listdir(args['model_dir'])
    y = None
    for model_name in model_list:
        wpath = os.path.join(args['model_dir'], model_name)
        model.load_weights(wpath)
        y0 = model.predict(X)
        if y is None:
            y = y0.copy()
        else:
            y += y0

    y = y / len(model_list)

    with open('result-test.csv', 'w') as fp:
        fp.write("Id,Category\n")
        for i in range(len(y)):
            id = np.argmax(y[i], axis=-1)
            if i != len(y)-1:
                fp.write('%d,%d\n' %(i, id))
            else:
                fp.write('%d,%d' % (i, id))

    for i in range(10):
        print((np.argmax(y, axis=-1) == i).mean())
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=str, help="Model ensembles dir", default=None)
    args = vars(parser.parse_args())
    main(args)
