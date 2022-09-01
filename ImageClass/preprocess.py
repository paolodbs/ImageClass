from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder


def data_augmentation(X_train):
    datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False)  # randomly flip images
    # (std, mean, and principal components if ZCA whitening is applied).
    X_train = datagen.fit(X_train)
    return X_train

def normalizing(X_train, X_test):
    X_train = X_train.astype('float32')/255
    X_test = X_test.astype('float32')/255
    return X_train, X_test

#need to change to categorizer:
def categorizing(y_train, y_test):
    encoder = OneHotEncoder(sparse = False)
    y_train_cat = encoder.fit_transform(y_train.reshape(-1,1))
    y_test_cat = encoder.fit_transform(y_test.reshape(-1,1))
    return y_train_cat, y_test_cat
