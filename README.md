# FER-2013

This repository is the result of a contest I won in my university. The objective was to train a model to detect emotions with [FER-2013](https://datarepository.wolframcloud.com/resources/FER-2013) dataset and create a simple program capable of finding faces in a bigger image and detect their emotions.

I fine-tuned a [VGG-Face model](https://www.robots.ox.ac.uk/~vgg/software/vgg_face/) with initial weights from [deepface](https://github.com/serengil/deepface) and replaced the head with a kNN classifier. It achieved an a accuracy of 69.1%. For the face detection I used [MTCNN](https://github.com/ipazc/mtcnn) and finally packed it in a simple command line Python app (detect_emotions.py) that shows or saves the given images with squares around the faces and the detected emotion for each one.
