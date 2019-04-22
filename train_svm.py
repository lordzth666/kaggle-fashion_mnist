import numpy as np
from src.preprocess import SplitTrainVal, EqualSplitTrainVal
from sklearn.svm import SVC
import pickle
import argparse

train_label_name = './dataset/train_labels.npy'
train_img_name = './dataset/train_images.npy'
test_img_name = './dataset/test_images.npy'

import argparse

def main(args):
    # Load the data
    X = np.load(train_img_name)
    y = np.load(train_label_name)
    _max = np.max(X)
    _min = np.min(X)

    # Normalize input to [0,1].
    X = (X - _min) / (_max - _min)
    indices = np.arange(np.shape(X)[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # Split train and validation data
    X_tr, y_tr, X_val, y_val = EqualSplitTrainVal(X, y, validation_split=0.1)

    print(y_tr)
    # Build the SVC models
    model = SVC(C=args.C, kernel=args.kernel, degree=args.degree, verbose=1,
                max_iter=10000, decision_function_shape='ovr')
    # Train the model
    model.fit(X_tr, y_tr)

    # Evaluate training and validation accuracy
    print("Training accuracy: %f"  %model.score(X_tr, y_tr))
    print("Validation accuracy: %f"  %model.score(X_val, y_val))

    # Dump model
    with open("models/svm_model/svm_%d" %args.C, 'wb') as fp:
        pickle.dump(model, fp)

if __name__ == "__main__":
    # Input the settings
    parser = argparse.ArgumentParser()
    parser.add_argument("--C", type=float, help="Penality term of SVM Classifier", required=True, default=1)
    parser.add_argument("--kernel", type=str, help="Kernel function", nargs='?', default='rbf')
    parser.add_argument("--degree", type=int, help="Degree of polynomial kernel", nargs='?', default=3)
    args = parser.parse_args()
    main(args)
