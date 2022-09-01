from tensorflow.keras import models
from tensorflow.keras import Sequential, layers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dropout,Dense,BatchNormalization,Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model

def initialize_model_1():
    model_pipe = Sequential([
    layers.Conv2D(32, (3,3), padding='same', input_shape=(32,32,3), activation="relu"),
    layers.MaxPool2D(pool_size=(2,2)),
    layers.Conv2D(64, (2,2), padding='same', activation="relu"),
    layers.MaxPool2D(pool_size=(2,2)),
    layers.Conv2D(64, (2,2), padding='same', activation="relu"),
    layers.MaxPool2D(pool_size=(2,2)),
    layers.Conv2D(64, (2,2), padding='same', activation="relu"),
    layers.MaxPool2D(pool_size=(2,2)),
    layers.Flatten(),
    layers.Dense(50, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(100, activation='softmax')
            ])
    model_pipe.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    return model_pipe


def initialize_model_2():
  model = Sequential()
  model.add(layers.Conv2D(input_shape=(32, 32, 3), kernel_size=(3, 3), padding='same', strides=(2, 2), filters=32))
  model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same'))
  model.add(layers.Conv2D(kernel_size=(3, 3), padding='same', strides=(2, 2), filters=40,activation='relu'))
  model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same'))
  model.add(layers.Flatten())
  model.add(layers.Dense(180, activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.Dropout(0.2))
  model.add(layers.Dense(150, activation='softmax'))
  model.add(layers.BatchNormalization())
  model.add(layers.Dropout(0.2))
  model.add(layers.Dense(128, activation='relu'))
  model.add(layers.BatchNormalization())
  model.add(layers.Dropout(0.2))
  model.add(layers.Dense(100, activation='softmax'))
  model.summary()
  model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])
  return model


def initialize_model_3():
    num_classes = 100
    resnet_model = ResNet50(weights='imagenet', include_top = False, input_shape = (32, 32, 3))
    optimizer = Adam()
    #resnet_model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
    x = resnet_model.output
    x = Flatten()(x)
    x = Dense(120, activation = 'relu')(x)
    predictions = Dense(num_classes, activation='softmax', kernel_initializer = 'random_uniform')(x)
    models = Model(inputs = resnet_model.input, outputs = predictions)
    for layer in resnet_model.layers:
        layer.trainable=True
    models.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
    return models

def save_model(model):
    model.save('natural_model')
