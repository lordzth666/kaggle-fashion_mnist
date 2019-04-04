import keras

def conv_bn_relu(X,
                 filters,
                 ksize=3,
                 regularizer_strength=1e-4):
    regularizer = keras.regularizers.l2(regularizer_strength)
    net = keras.layers.Conv2D(filters=filters,
                              kernel_size=(ksize, ksize),
                              strides=(1, 1),
                              kernel_regularizer=regularizer,
                              padding='SAME',
                              activation="linear",
                              )(X)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.ReLU()(net)
    return net

def residual_block(X,
                   filters,
                   regularizer_strength=1e-4):
    regularizer = keras.regularizers.l2(regularizer_strength)
    net = conv_bn_relu(X, filters=filters, regularizer_strength=regularizer_strength)
    net = conv_bn_relu(net, filters=filters, regularizer_strength=regularizer_strength)
    if X.get_shape()[-1] != net.get_shape()[-1]:
        X_id = keras.layers.Conv2D(filters=filters,
                                   kernel_size=(1, 1),
                                   strides=(1, 1),
                                   kernel_regularizer=regularizer,
                                   padding='SAME',
                                   activation='linear')(X)
        net = keras.layers.Add()([net, X_id])
    else:
        net = keras.layers.Add()([net, X])
    return net



def ResNet_18(input_shape=(28, 28, 1),
              regularizer_strength=1e-4,
              scale=1.0):
    regularizer = keras.regularizers.l2(regularizer_strength)
    net_in = keras.layers.Input(shape=input_shape)
    net = conv_bn_relu(net_in, ksize=5, filters=int(64*scale), regularizer_strength=regularizer_strength)
    print(net.get_shape())
    net = keras.layers.MaxPool2D(strides=2, pool_size=2, padding='SAME')(net)
    net = residual_block(net, filters=int(128*scale), regularizer_strength=regularizer_strength)
    net = residual_block(net, filters=int(128*scale), regularizer_strength=regularizer_strength)
    net = keras.layers.MaxPool2D(strides=2, pool_size=2, padding='SAME')(net)
    print(net.get_shape())
    net = residual_block(net, filters=int(256*scale), regularizer_strength=regularizer_strength)
    net = residual_block(net, filters=int(256*scale), regularizer_strength=regularizer_strength)
    net = residual_block(net, filters=int(256*scale), regularizer_strength=regularizer_strength)
    net = residual_block(net, filters=int(256*scale), regularizer_strength=regularizer_strength)
    #net = keras.layers.MaxPool2D(strides=2, pool_size=2, padding='SAME')(net)
    print(net.get_shape())
    net = residual_block(net, filters=int(512*scale), regularizer_strength=regularizer_strength)
    net = residual_block(net, filters=int(512*scale), regularizer_strength=regularizer_strength)
    net = residual_block(net, filters=int(512*scale), regularizer_strength=regularizer_strength)
    net = residual_block(net, filters=int(512*scale), regularizer_strength=regularizer_strength)
    net = residual_block(net, filters=int(512*scale), regularizer_strength=regularizer_strength)
    net = residual_block(net, filters=int(512*scale), regularizer_strength=regularizer_strength)

    net = keras.layers.Flatten()(net)

    net = keras.layers.Dense(10, activation='softmax')(net)

    return keras.models.Model(inputs=net_in, outputs=net)


def LeNet_5(input_shape=(28, 28, 1),
            regularizer_strength=1e-4):
    """
    Define a LeNet-5 model
    :param inputs:
    :return: a Keras model
    """
    regularizer = keras.regularizers.l2(regularizer_strength)
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(filters=32,
                                  kernel_size=(5, 5),
                                  strides=(1, 1),
                                  kernel_regularizer=regularizer,
                                  padding='SAME',
                                  activation="linear",
                                  input_shape=input_shape
                                  ))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(keras.layers.Conv2D(filters=64,
                                  kernel_size=(3, 3),
                                  strides=(1, 1),
                                  kernel_regularizer=regularizer,
                                  padding='SAME',
                                  activation="linear",
                                  ))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())

    model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(120, activation='linear',
                                 kernel_regularizer=regularizer,
                                 ))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())

    model.add(keras.layers.Dense(84, activation='linear',
                                 kernel_regularizer=regularizer,
                                 ))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())

    model.add(keras.layers.Dense(10, activation='softmax',
                                 ))
    return model



def VGG_9(input_shape=(28, 28, 1),
            regularizer_strength=1e-4):
    """
    Define a LeNet-5 model
    :param inputs:
    :return: a Keras model
    """
    regularizer = keras.regularizers.l2(regularizer_strength)
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(filters=32,
                                  kernel_size=(3, 3),
                                  strides=(1, 1),
                                  kernel_regularizer=regularizer,
                                  padding='SAME',
                                  activation="linear",
                                  input_shape=input_shape
                                  ))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())

    model.add(keras.layers.Conv2D(filters=32,
                                  kernel_size=(3, 3),
                                  strides=(1, 1),
                                  kernel_regularizer=regularizer,
                                  padding='SAME',
                                  activation="linear",
                                  ))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())

    model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(keras.layers.Conv2D(filters=64,
                                  kernel_size=(3, 3),
                                  strides=(1, 1),
                                  kernel_regularizer=regularizer,
                                  padding='SAME',
                                  activation="linear",
                                  ))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())


    model.add(keras.layers.Conv2D(filters=64,
                                  kernel_size=(3, 3),
                                  strides=(1, 1),
                                  kernel_regularizer=regularizer,
                                  padding='SAME',
                                  activation="linear",
                                  ))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())

    model.add(keras.layers.Conv2D(filters=64,
                                  kernel_size=(3, 3),
                                  strides=(1, 1),
                                  kernel_regularizer=regularizer,
                                  padding='SAME',
                                  activation="linear",
                                  ))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())

    model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(keras.layers.Conv2D(filters=128,
                                  kernel_size=(3, 3),
                                  strides=(1, 1),
                                  kernel_regularizer=regularizer,
                                  padding='SAME',
                                  activation="linear",
                                  ))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())

    model.add(keras.layers.Conv2D(filters=128,
                                  kernel_size=(3, 3),
                                  strides=(1, 1),
                                  kernel_regularizer=regularizer,
                                  padding='SAME',
                                  activation="linear",
                                  ))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())

    model.add(keras.layers.Conv2D(filters=128,
                                  kernel_size=(3, 3),
                                  strides=(1, 1),
                                  kernel_regularizer=regularizer,
                                  padding='SAME',
                                  activation="linear",
                                  ))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())

    model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(1024, activation='relu',
                                 kernel_regularizer=regularizer,
                                 ))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())

    model.add(keras.layers.Dense(1024, activation='relu',
                                 kernel_regularizer=regularizer,
                                 ))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.ReLU())

    model.add(keras.layers.Dense(10, activation='softmax',
                                 ))
    return model

