import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import load_img, smart_resize
from mtcnn import MTCNN
from models import vgg_face_model as vggface
from models import scratch_model as scratch
from PIL import Image, ImageDraw, UnidentifiedImageError


_EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']


class EmotionDetector:
    
    def __init__(self, model='vggface_knn', color_mode=None):
        if isinstance(model, str):
            if model == 'vggface':
                self.model = vggface.load_model()
                color_mode = 'rgb'
            elif model == 'scratch':
                self.model = scratch.load_model()
                color_mode = 'grayscale'
            else:
                self.model = vggface.VGGFaceKNN()
                color_mode = 'rgb'
        else:
            self.model = model
        self.target_size = self.model.input_shape[1:3]
        self.color_mode = color_mode
    
    def read_images(self, path):
        if os.path.isdir(path):
            filenames = [os.path.join(path, filename)
                         for _, _, filenames in os.walk(path)
                         for filename in filenames]
        else:
            filenames = [path]
        
        imgs = []
        for filename in filenames:
            try:
                imgs.append(np.array(Image.open(filename).convert('RGB')))
            except UnidentifiedImageError:
                # Ignore files that are not images
                print(f"[WARNING] ignoring {filename}: unrecognized image type")
        return imgs
    
    def detect_emotion(self, paths, show=False, save=None, verbose=False):
        def cut(img, box):
            x, y, w, h = box
            x0 = max(x, 0)
            y0 = max(y, 0)
            return img[y0:y+h, x0:x+w, :]
        
        # https://stackoverflow.com/questions/46836358/keras-rgb-to-grayscale/51879084
        def tograyscale(x):
            # x has shape (batch, width, height, channels)
            img = ((.21 * x[:, :, :, :1]) +
                   (.72 * x[:, :, :, 1:2]) +
                   (.07 * x[:, :, :, -1:])).reshape(*x.shape[:3])
            if self.color_mode == 'grayscale':
                print(img.shape)
                return np.expand_dims(img, axis=-1) if len(img.shape) == 3 else img
            return np.stack((img,) * 3, axis=-1)

        # ----------------------------
        # Load images
        # ----------------------------
        
        if isinstance(paths, list):
            imgs = []
            for path in paths:
                temp = self.read_images(path)
                imgs += temp
        elif isinstance(paths, str):
            imgs = self.read_images(paths)
        
        # ----------------------------
        # Detect faces
        # ----------------------------

        detector = MTCNN()
        
        # All faces
        faces = []
        # Number of faces in each image
        n_faces = []
        for img in imgs:
            temp = detector.detect_faces(img)
            faces += temp
            n_faces.append(len(temp))
        
        if verbose:
            print(f"Found {len(imgs)} image(s) containing {len(faces)} faces")
        
        if not faces or all(not f for f in faces):
            return []
        
        # ----------------------------
        # Get all boxes
        # ----------------------------
        
        boxes = []
        k = 0
        for img, n in zip(imgs, n_faces):
            faces_ = faces[k:k+n]
            for face in faces_:
                boxes.append(smart_resize(cut(img, face['box']), size=self.target_size))
            k += n
        
        boxes = tograyscale(np.array(boxes)) / 255
        
        # ----------------------------
        # Predict emotions
        # ----------------------------
        
        predictions = self.model.predict(boxes)
        
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            emotions = np.argmax(predictions, axis=-1)
        else:
            emotions = predictions
        names = [_EMOTIONS[e] for e in emotions]
        
        # ----------------------------
        # Show results
        # ----------------------------
        
        if show or save is not None:
            self._make_images(imgs, n_faces, faces, names, show, save)
        
        k = 0
        temp1 = []
        temp2 = []
        for n in n_faces:
            temp1.append(faces[k:k+n])
            temp2.append(names[k:k+n])
            k += n
        faces = temp1
        names = temp2
            
        return list(zip(faces, names))
    
    @staticmethod
    def _make_images(imgs, n_faces, faces, names, show, save):
        def random_color():
            return np.random.choice(['red', 'green', 'blue', 'orange',
                                     'purple', 'pink', 'white'])
        
        if save is not None and not os.path.isdir(save):
            os.mkdir(save)
        
        pad_size = len(str(len(imgs)))
        
        k = 0
        for index, (img, n) in enumerate(zip(imgs, n_faces)):
            image = Image.fromarray(img)
            draw = ImageDraw.Draw(image)
            faces_ = faces[k:k+n]
            names_ = names[k:k+n]
            for face, name in zip(faces_, names_):
                x0, y0, w, h = face['box']
                text_x = x0 + 4
                text_y = y0 - 10 if y0 >= 10 else max(0, y0)
                color = random_color()
                draw.rectangle([x0, y0, x0 + w, y0 + h], outline=color, width=2)
                draw.text((text_x, text_y), name, fill=color)
            
            if save is not None:
                image.save(os.path.join(save, f'{index}'.zfill(pad_size) + '.jpg'))
            
            if show:
                image.show()
                if index != 0 and index % 4 == 0:
                    input("Press enter to continue...")
            
            k += n
