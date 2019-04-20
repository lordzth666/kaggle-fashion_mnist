from sklearn.svm import SVC
import numpy as np
from src.preprocess import EqualSplitTrainVal

# Load the training and testing data
train_label_name = './dataset/train_labels.npy'
train_img_name = './dataset/train_images.npy'
test_img_name = './dataset/test_images.npy'
X = np.load(train_img_name)
y = np.load(train_label_name)

# Reshape the data to 3D image shape
X = np.reshape(X, [-1, 28, 28, 1])

# Reduce mean and variance. _mean and _var are precalculated.
X = (X - _mean) / _var

# Split dataset
X_tr, y_tr, X_val, y_val = EqualSplitTrainVal(X, y, validation_split=0.1)

# Define the SVM Classifier SVC with configuration args
model = SVC(C=args.C, kernel=args.kernel, degree=args.degree, verbose=1,
            max_iter=10000, decision_function_shape='ovr')

# Fit the model
model.fit(X_tr, y_tr)
# After that, we can collect metrics like accuracy, probs etc.