import numpy as np
from src.preprocess import SplitTrainVal
import math
import autokeras

train_label_name = './dataset/train_labels.npy'
train_img_name = './dataset/train_images.npy'
test_img_name = './dataset/test_images.npy'

def main():
    X = np.load(train_img_name)
    y = np.load(train_label_name)
    X = np.reshape(X, [-1, 28, 28, 1])

    _mean = np.mean(X, axis=(0, 1, 2))
    _std = np.std(X, axis=(0, 1, 2))

    print("mean: %f" %_mean)
    print("std: %f" %_std)

    with open("stats.txt", 'w') as fp:
        fp.write('%f\n' %_mean)
        fp.write('%f\n' %_std)

    X = (X-_mean) / _std

    X_tr, y_tr, X_val, y_val = SplitTrainVal(X, y, validation_split=0.2)
    model = autokeras.ImageClassifier(verbose=True)
    model.fit(X_tr, y_tr)



if __name__ == "__main__":
    main()
