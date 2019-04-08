import numpy as np
import cv2

def SplitTrainVal(X, y, validation_split=.2):
    np.random.seed()
    N = np.shape(X)[0]
    indices = np.random.permutation(N)
    train_idx = indices[0:int(N*(1-validation_split))]
    val_idx = indices[int(N*(1-validation_split)):]
    X_tr = X[train_idx]
    y_tr = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]
    return X_tr, y_tr, X_val, y_val


def Randomcrop_and_pad(X, padding=4, training=True):
    if not training:
        return X
    X_batch = np.zeros_like(X)
    img_shape = np.shape(X)[1:]
    padded_img_shape = list(img_shape)
    padded_img_shape[0] += 2 * padding
    padded_img_shape[1] += 2 * padding
    for i in range(len(X)):
        X_batch_pad = np.zeros(padded_img_shape)
        X_batch_pad[padding:padding+img_shape[0], padding:padding+img_shape[1], :] = X[i]
        idx1 = np.random.randint(padding*2)
        idx2 = np.random.randint(padding*2)
        X_batch[i] = X_batch_pad[idx1:idx1+img_shape[0], idx2:idx2+img_shape[1], :]
        if np.random.rand() > .5:
            X_batch[i] = np.fliplr(X_batch[i])
        if np.random.rand() > .5:
            X_batch[i] = np.flipud(X_batch[i])
    return X_batch


def EqualSplitTrainVal(X, y, validation_split):

    n = np.shape(X)[0]
    num_classes = np.max(y) - np.min(y) + 1
    num_per_classes = int(n * validation_split / num_classes)
    print(num_per_classes)
    val_indices = []
    tr_indices = []
    for i in range(num_classes):
        indices = np.where(y == i)[0]
        np.random.shuffle(indices)
        val_indices += list(indices[:num_per_classes])
        tr_indices += list(indices[num_per_classes:])

    tr_indices = np.asarray(tr_indices, dtype=np.int32)
    val_indices = np.asarray(val_indices, dtype=np.int32)

    np.random.shuffle(tr_indices)
    np.random.shuffle(val_indices)

    X_val = X[val_indices]
    y_val = y[val_indices]
    X_tr = X[tr_indices]
    y_tr = y[tr_indices]
    print(np.shape(X_tr))
    print(np.shape(X_val))
    return X_tr, y_tr, X_val, y_val
