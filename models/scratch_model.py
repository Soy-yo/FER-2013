from tensorflow.keras.layers import (
    Flatten, Dense, Dropout, Conv2D, MaxPooling2D
)
from tensorflow.keras.models import Sequential


DEFAULT_WEIGHTS_FILE = 'weights/emotions_scratch_v5.h5'


def base_model():
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(48, 48, 1)))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(.1))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(.1))

    model.add(Dense(7, activation='softmax'))
    
    return model
    

def load_model(weights_file=DEFAULT_WEIGHTS_FILE):
    model = base_model()
    
    if weights_file is not None:
        model.load_weights(weights_file)
    
    return model
