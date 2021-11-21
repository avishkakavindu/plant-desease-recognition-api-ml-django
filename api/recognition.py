import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dropout
from keras.layers import Flatten
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from sklearn.metrics import confusion_matrix
import cv2
from glob import glob

scale = 70
seed = 7


class plant_leaf_disease_recognition:

    def __init__(self):

        # print("Recognise Image")
        self.path_to_images = '{}/*/*.jfif'.format(os.path.join(os.getcwd(), 'train'))
        self.images = glob(self.path_to_images)
        print(self.images)

        self.trainingset = []
        self.traininglabels = []
        self.new_train = []
        self.sets = []
        self.num = len(self.images)
        self.x_train = []
        self.x_test = []
        self.y_train = []
        self.y_test = []
        self.model = Sequential()

    def preprocess_images(self):
        count = 1
        for i in self.images:
            self.trainingset.append(cv2.resize(cv2.imread(i), (scale, scale)))
            self.traininglabels.append(i.split('/')[-2])
            count = count + 1
        self.traininglabels = pd.DataFrame(self.traininglabels)
        self.trainingset = np.asarray(self.trainingset)

        for i in self.trainingset:
            blurr = cv2.GaussianBlur(i, (5, 5), 0)
            hsv = cv2.cvtColor(blurr, cv2.COLOR_BGR2HSV)

            lower = (25, 40, 50)
            upper = (75, 255, 255)
            mask = cv2.inRange(hsv, lower, upper)
            struc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, struc)
            boolean = mask > 0
            new = np.zeros_like(i, np.uint8)
            new[boolean] = i[boolean]
            self.new_train.append(new)

        self.new_train = np.asarray(self.new_train)
        self.labels = preprocessing.LabelEncoder()
        self.labels.fit(self.traininglabels[0])

        self.encodedlabels = self.labels.transform(self.traininglabels[0])
        clearalllabels = np_utils.to_categorical(self.encodedlabels)
        self.classes = clearalllabels.shape[1]
        print('Classes' + str(self.labels.classes_))

        self.new_train = self.new_train / 255
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.new_train, clearalllabels,
                                                                                test_size=0.1, random_state=seed,
                                                                                stratify=clearalllabels)

    def create_model(self):

        generator = ImageDataGenerator(rotation_range=180, zoom_range=0.1, width_shift_range=0.1,
                                       height_shift_range=0.1, horizontal_flip=True, vertical_flip=True)
        generator.fit(self.x_train)

        np.random.seed(seed)

        self.model.add(Conv2D(filters=64, kernel_size=(5, 5), input_shape=(scale, scale, 3), activation='relu'))
        self.model.add(BatchNormalization(axis=3))
        self.model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(BatchNormalization(axis=3))
        self.model.add(Dropout(0.1))
        self.model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))
        self.model.add(BatchNormalization(axis=3))
        self.model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(BatchNormalization(axis=3))
        self.model.add(Dropout(0.1))
        self.model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))
        self.model.add(BatchNormalization(axis=3))
        self.model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(BatchNormalization(axis=3))
        self.model.add(Dropout(0.1))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(256, activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.classes, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()

    def train_model(self):

        self.preprocess_images()
        self.create_model()

        history = self.model.fit(self.x_train, self.y_train, epochs=100)

        plt.plot(history.history['loss'])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.show()

        pred_labels = self.model.predict(self.x_test)
        print(pred_labels.shape)
        acc = self.model.evaluate(self.x_test, self.y_test)

        self.save_model()

    def save_model(self):

        keras_file = "model.h5"
        self.model.save(keras_file)

        # Convert to TensorFlow Lite model.
        # converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file)
        # tflite_model = converter.convert()
        # open("converted_model.tflite", "wb").write(tflite_model)

    def predict(self, img_path):

        traininglabels = ['Helthy Leaf', 'Anthracnose', 'Bacterial_blight', 'Cercospora_Leaf_spot', 'Powdery_mildew']

        img = cv2.imread(img_path)
        img = cv2.resize(img, (70, 70))
        inputset = []
        inputset.append(img)

        new_img = []
        for i in inputset:
            blurr = cv2.GaussianBlur(i, (5, 5), 0)
            hsv = cv2.cvtColor(blurr, cv2.COLOR_BGR2HSV)
            # GREEN PARAMETERS
            lower = (25, 40, 50)
            upper = (75, 255, 255)
            mask = cv2.inRange(hsv, lower, upper)
            struc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, struc)
            boolean = mask > 0
            new = np.zeros_like(i, np.uint8)
            new[boolean] = i[boolean]
            new_img.append(new)

        new_img = np.asarray(new_img)
        new_img = new_img / 255

        loaded_model = tf.keras.models.load_model('api/model/model.h5')
        result = loaded_model.predict(new_img)
        # sort indexes of predictions high to low
        sorted_result = (-result[0]).argsort()
        return traininglabels[sorted_result[0]]


# test = plant_leaf_disease_recognition()
# test.train_model()

# print(test.predict("/Users/roshanwithanage/Desktop/SLIIT JUNIOR PROJECTS/2021/Kanishka/Plant/a.jfif"))
