import sys
import util
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
from models import vgg_face_model as vgg


def main():
    if len(sys.argv) != 2:
        print('Expected exactly one argument: data path')
        exit(1)
    
    path = sys.argv[1]
    
    model = vgg.VGGFaceKNN()
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Expects a path containing one directory for each emotion
    test_generator = test_datagen.flow_from_directory(
        path,
        target_size=model.input_shape[1:3],
        batch_size=64,
        class_mode='categorical',
        color_mode='rgb',
        shuffle=False
    )
    
    print("Predicting... This may take a few minutes...")
    
    predictions = model.predict(test_generator)
    labels = test_generator.labels
    classes = list(test_generator.class_indices.keys())
    
    acc = (predictions == labels).sum() / len(labels)
    cm = confusion_matrix(labels, predictions)
    
    print(f"Precision: {acc}")
    util.plot_confusion_matrix(cm, classes, 'Class predictions and recall')
    print(classification_report(labels, predictions, target_names=classes))


if __name__ == '__main__':
    main()
