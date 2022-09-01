import numpy as np
from google.colab import drive


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

def create_X_y():
    filetrain = '/content/drive/MyDrive/data2/train'
    filetest = '/content/drive/MyDrive/data2/test'
    drive.mount('/content/drive' , force_remount = True)
    train = unpickle(file)
    test = unpickle(filetest)
    X_train = train[b'data']
    X_train = X_train.reshape(len(X_train),3,32,32).transpose(0,2,3,1)
    X_test = test[b'data']
    X_test = X_test.reshape(len(X_test),3,32,32).transpose(0,2,3,1)
    y_train = np.array(train[b'fine_labels'])
    y_test = np.array(test[b'fine_labels'])
    return X_train, X_test, y_train, y_test


def create_class_dic():
    metadata_path = '/content/drive/MyDrive/data2/meta'
    metadata = unpickle(metadata_path)
    class_dict = {key : value.decode() for key , value in dict(list(enumerate(metadata[b'fine_label_names']))).items()}
    return class_dict

def create_labels(class_dict):
    labels = list(class_dict.values())
    return labels
