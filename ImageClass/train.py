from tensorflow.keras import models
from tensorflow.keras import callbacks
#from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras import Model
from ImageClass.architecture import initialize_model_1, initialize_model_2, initialize_model_3

es = callbacks.EarlyStopping(patience=30, restore_best_weights=True)

def train_model_1(X_train, y_train_cat):
    model = initialize_model_1()
    model.fit(X_train, y_train_cat,
          batch_size=16,
          epochs=30,
          validation_split=0.3,
          callbacks=[es],
          verbose=1)
    return model

def train_model_2(X_train, y_train_cat):
    model = initialize_model_2()
    model.fit(X_train, y_train_cat,
          batch_size=16,
          epochs=30,
          validation_split=0.3,
          callbacks=[es],
          verbose=1)
    return model

def train_model_3(X_train, y_train_cat , X_test , y_test_cat):
    model = initialize_model_3()
    batch_size = 20
    maxepoches = 50
    shape = (32,32,3)
    hist = model.fit_generator(datagen.flow(X_train, y_train_cat,
                                 batch_size=batch_size),
                                steps_per_epoch = X_train.shape[0] // batch_size,
                                epochs = maxepoches,
                                validation_data = (X_test, y_test_cat))
    model.save_weights('cifar100resnet.h5')
    return model
