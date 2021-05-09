import numpy as np
import os

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Flatten, Conv2D, ZeroPadding2D, Convolution2D, MaxPooling2D,
    Dropout, Activation
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer


TOP_WEIGHTS = 'weights/vgg_face_weights.h5'
DEFAULT_WEIGHTS_FILE = 'weights/emotions_vggface_v3.h5'


def remove_top(model, n):
    return Model(inputs=model.layers[0].input,
                 outputs=model.layers[-n-1].output)


def base_model():

    model = Sequential()
    
    model.add(ZeroPadding2D((1, 1),input_shape=(224, 224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
     
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
     
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
     
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
     
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
     
    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    
    return model


def load_model(weights_file=DEFAULT_WEIGHTS_FILE):

    model = Sequential()

    base = base_model()
    base.load_weights(TOP_WEIGHTS)
    base = remove_top(base, 7)

    for layer in base.layers[:-7]:
        layer.trainable = False

    model.add(base)

    model.add(Conv2D(256, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(256, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(7, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    
    if weights_file is not None:
        model.load_weights(weights_file)
    
    return model


class VGGFaceKNN:

    def __init__(self, k=35):
        model = load_model()
        self.model = Model(inputs=model.input, outputs=model.layers[-2].output)
        self.normalizer = Normalizer('l2')
        x_train = np.load('arrays/x_repr_vggface_v3.npy')
        y_train = np.load('arrays/y_repr_vggface_v3.npy')
        self.knn = KNeighborsClassifier(k, weights='distance').fit(x_train, y_train)
    
    @property
    def input_shape(self):
        return self.model.input_shape
    
    def predict(self, x):
        y = self.model.predict(x)
        y = self.normalizer.fit_transform(y)
        return self.knn.predict(y)
    
    def score(self, x, y=None):
        if y is None:
            # Assuming x is a generator
            y = x.labels
        p = self.predict(x)
        return (p == y).sum() / len(y)
    
    def evaluate(self, x, y=None):
        return self.score(x, y)
    