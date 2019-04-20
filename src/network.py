import keras

# Network Model Zoo.
# Supports: Wide ResNet
#           LeNet_5
#           VGG_10


def conv_bn_relu(X,
                 filters,
                 ksize=3,
                 strides=1,
                 regularizer_strength=1e-4):
    regularizer = keras.regularizers.l2(regularizer_strength)
    net = keras.layers.Conv2D(filters=filters,
                              kernel_size=(ksize, ksize),
                              strides=strides,
                              kernel_regularizer=regularizer,
                              padding='SAME',
                              activation="linear",
                              kernel_initializer=keras.initializers.he_normal(),
                              )(X)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.ReLU()(net)
    return net

def residual_block(X,
                   filters,
                   regularizer_strength=1e-4,
                   dropout=0.0,
                   strides=1):
    regularizer = keras.regularizers.l2(regularizer_strength)
    net = keras.layers.Conv2D(filters=filters,
                              kernel_size=(3, 3),
                              strides=(strides, strides),
                              kernel_regularizer=regularizer,
                              padding='SAME',
                              activation="linear",
                              kernel_initializer=keras.initializers.he_normal(),
                              )(X)
    net = keras.layers.BatchNormalization()(net)
    net = keras.layers.ReLU()(net)

    net = keras.layers.Dropout(rate=dropout)(net)

    net = keras.layers.Conv2D(filters=filters,
                              kernel_size=(3, 3),
                              strides=(1, 1),
                              kernel_regularizer=regularizer,
                              padding='SAME',
                              activation="linear",
                              kernel_initializer=keras.initializers.he_normal(),
                              )(net)
    net = keras.layers.BatchNormalization()(net)

    if X.get_shape()[-1] != net.get_shape()[-1]:
        X_id = keras.layers.Conv2D(filters=filters,
                                   kernel_size=(1, 1),
                                   strides=(strides, strides),
                                   kernel_regularizer=regularizer,
                                   padding='SAME',
                                   activation='linear',
                                   kernel_initializer=keras.initializers.he_normal())(X)
        X_id = keras.layers.BatchNormalization()(X_id)

        net = keras.layers.Add()([net, X_id])
    else:
        net = keras.layers.Add()([net, X])

    net = keras.layers.ReLU()(net)
    return net


def Wide_ResNet(input_shape=(28, 28, 1),
                regularizer_strength=1e-4,
                n=2,
                k=2,
                dropout=.1):

    print(regularizer_strength)

    net_in = keras.layers.Input(shape=input_shape)
    net = conv_bn_relu(net_in, ksize=3, filters=16, strides=2, regularizer_strength=regularizer_strength)

    for i in range(n):
        strides = 1 + (i == 0)
        net = residual_block(net, filters=16*k, regularizer_strength=regularizer_strength, dropout=dropout, strides=strides)

    for i in range(n):
        strides = 1 + (i == 0)
        net = residual_block(net, filters=32*k, regularizer_strength=regularizer_strength, dropout=dropout, strides=strides)

    for i in range(n):
        strides = 1 + (i == 0)
        net = residual_block(net, filters=64*k, regularizer_strength=regularizer_strength, dropout=dropout, strides=strides)

    net = keras.layers.GlobalAvgPool2D()(net)

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



# VGG10: The model we used for evaluation.
def VGG_10(input_shape=(28, 28, 1),
           regularizer_strength=1e-4,
           dropout=0.5,
           batchnorm=True):
    initializer = keras.initializers.he_normal()
    regularizer = keras.regularizers.l2(regularizer_strength)
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(filters=32,
                                  kernel_size=(3, 3),
                                  strides=(1, 1),
                                  kernel_initializer=initializer,
                                  kernel_regularizer=regularizer,
                                  padding='SAME',
                                  activation="linear",
                                  input_shape=input_shape
                                  ))
    if batchnorm:
        model.add(keras.layers.BatchNormalization(momentum=.997))
    model.add(keras.layers.ReLU())


    model.add(keras.layers.Conv2D(filters=32,
                                  kernel_size=(3, 3),
                                  strides=(1, 1),
                                  kernel_initializer=initializer,
                                  kernel_regularizer=regularizer,
                                  padding='SAME',
                                  activation="linear",
                                  input_shape=input_shape
                                  ))
    if batchnorm:
        model.add(keras.layers.BatchNormalization(momentum=.997))

    model.add(keras.layers.ReLU())

    model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(keras.layers.Conv2D(filters=64,
                                  kernel_size=(3, 3),
                                  strides=(1, 1),
                                  kernel_initializer=initializer,
                                  kernel_regularizer=regularizer,
                                  padding='SAME',
                                  activation="linear",
                                  ))
    if batchnorm:
        model.add(keras.layers.BatchNormalization(momentum=.997))
    model.add(keras.layers.ReLU())


    model.add(keras.layers.Conv2D(filters=64,
                                  kernel_size=(3, 3),
                                  strides=(1, 1),
                                  kernel_initializer=initializer,
                                  kernel_regularizer=regularizer,
                                  padding='SAME',
                                  activation="linear",
                                  ))
    if batchnorm:
        model.add(keras.layers.BatchNormalization(momentum=.997))
    model.add(keras.layers.ReLU())

    model.add(keras.layers.Conv2D(filters=64,
                                  kernel_size=(3, 3),
                                  strides=(1, 1),
                                  kernel_initializer=initializer,
                                  kernel_regularizer=regularizer,
                                  padding='SAME',
                                  activation="linear",
                                  ))
    if batchnorm:
        model.add(keras.layers.BatchNormalization(momentum=.997))
    model.add(keras.layers.ReLU())

    model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(keras.layers.Conv2D(filters=128,
                                  kernel_size=(3, 3),
                                  strides=(1, 1),
                                  kernel_initializer=initializer,
                                  kernel_regularizer=regularizer,
                                  padding='SAME',
                                  activation="linear",
                                  ))
    if batchnorm:
        model.add(keras.layers.BatchNormalization(momentum=.997))
    model.add(keras.layers.ReLU())


    model.add(keras.layers.Conv2D(filters=128,
                                  kernel_size=(3, 3),
                                  strides=(1, 1),
                                  kernel_initializer=initializer,
                                  kernel_regularizer=regularizer,
                                  padding='SAME',
                                  activation="linear",
                                  ))
    if batchnorm:
        model.add(keras.layers.BatchNormalization(momentum=.997))
    model.add(keras.layers.ReLU())

    model.add(keras.layers.Conv2D(filters=128,
                                  kernel_size=(3, 3),
                                  strides=(1, 1),
                                  kernel_initializer=initializer,
                                  kernel_regularizer=regularizer,
                                  padding='SAME',
                                  activation="linear",
                                  ))
    if batchnorm:
        model.add(keras.layers.BatchNormalization(momentum=.997))
    model.add(keras.layers.ReLU())

    model.add(keras.layers.GlobalAvgPool2D())

    model.add(keras.layers.Dense(512, activation='linear',
                                 kernel_regularizer=regularizer,
                                 ))
    if batchnorm:
        model.add(keras.layers.BatchNormalization(momentum=.997))
    model.add(keras.layers.ReLU())
    model.add(keras.layers.Dropout(dropout))

    model.add(keras.layers.Dense(10, activation='softmax',
                                 ))
    return model

