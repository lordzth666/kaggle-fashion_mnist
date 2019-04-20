import numpy as np
from src.preprocess import SplitTrainVal
from src.network import *
import keras


train_label_name = './dataset/train_labels.npy'
train_img_name = './dataset/train_images.npy'
test_img_name = './dataset/test_images.npy'

def main():
    # Load dataset
    X = np.load(test_img_name)
    X = np.reshape(X, [-1, 28, 28, 1])
    X = (X - 0.28603) / 0.352901

    # Define the model
    model = VGG_10(input_shape=(28, 28, 1),
                   regularizer_strength=2e-4,
                   dropout=0.0)
    # Compile the model
    model.compile(optimizer=keras.optimizers.sgd(lr=1e-2, momentum=.9),
                  loss=keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    # Print the model summary
    model.summary()
    # Load the pretrained weights
    model.load_weights('./models/model.ckpt')
    # Predict the id
    y = model.predict(X)
    # Open the target file and write to target file
    with open('result-test.csv', 'w') as fp:
        fp.write("Id,Category\n")
        for i in range(len(y)):
            id = np.argmax(y[i], axis=-1)
            if i != len(y)-1:
                fp.write('%d,%d\n' %(i, id))
            else:
                fp.write('%d,%d' % (i, id))
if __name__ == "__main__":
    main()
