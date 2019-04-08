import numpy as np
from src.preprocess import SplitTrainVal
from src.network import *
import keras


train_label_name = './dataset/train_labels.npy'
train_img_name = './dataset/train_images.npy'
test_img_name = './dataset/test_images.npy'

def main():
    X = np.load(test_img_name)
    X = np.reshape(X, [-1, 28, 28, 1])
    X = (X - 0.28603) / 0.352901

    model = VGG_9(input_shape=(28, 28, 1),
                   regularizer_strength=2e-4)
    model.compile(optimizer=keras.optimizers.sgd(lr=1e-2, momentum=.9),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    model.summary()
    model.load_weights('./models/model.ckpt')
    y = model.predict(X)
    with open('result-test.csv', 'w') as fp:
        fp.write("Id,Category\n")
        for i in range(len(y)):
            id = np.argmax(y[i], axis=-1)
            if i != len(y)-1:
                fp.write('%d,%d\n' %(i, id))
            else:
                fp.write('%d,%d' % (i, id))

    for i in range(10):
        print((np.argmax(y) == i).mean())
if __name__ == "__main__":
    main()
