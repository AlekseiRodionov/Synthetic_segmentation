import os

import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image


def load_dataset(path):
    """
    Loads the dataset. It includes original images, as well as masks that can be used
    to highlight objects in the photo.
    Args:
        path (str): The path to the dataset. The specified path should contain a folder with images
                    called "Photos_with_rocks", as well as a folder with masks called "Photos_with_masks"
    Returns:
        tuple: a tuple with np.arrays containing images and masks to them in the form of 2D np.arrays.
    """
    X_data, y_data = [], []
    for name in os.listdir(path + '/Photos_with_rocks'):
        img = Image.open(path + '/Photos_with_rocks/' + name).convert('RGB')
        img_array = np.array(img) / 255.0
        X_data.append(img_array)
    for name in os.listdir(path + '/Photos_with_masks'):
        mask = Image.open(path + '/Photos_with_masks/' + name).convert('L')
        mask_array = np.array(mask)[..., None] / 255.0
        mask_array[mask_array > 0.2] = 1.0
        y_data.append(mask_array)
    return np.array(X_data), np.array(y_data)


class Model(tf.keras.Model):
    """
    Class for defining the Fully-convolutional network (FCN). Standard methods are used.

    Methods:
        call(x):
            Determines the calculations. It is used both in training and in prediction.
    """

    def __init__(self):
        """
        Defines the architecture of the neural network.
        Args:
            None.
        Return:
            None.
        """
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')
        self.conv3 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu')
        self.conv5 = tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu')
        self.conv6 = tf.keras.layers.Conv2D(1, (3, 3), padding='same', activation=None)
        self.pool = tf.keras.layers.MaxPool2D((2, 2))

    def call(self, x):
        """
        Determines the calculations. It is used both in training and in prediction.
        Args:
            x (np.array): Training data. In this case, a 2D array representing an image.
        Return:
            out (np.array): The output value. A 2D array filled with real numbers in the range from 0 to 1, showing
                            the probability that there is an object in a particular pixel. In fact, the 2D output array
                            is the predicted image mask.
        """
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.pool(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.pool(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = tf.image.resize(out, (x.shape[1], x.shape[2]), tf.image.ResizeMethod.BILINEAR)
        out = tf.nn.sigmoid(out)
        return out


if __name__ == '__main__':
    X_data, y_data = load_dataset('Вырезанные_фото_камней')
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
    model = Model()
    checkpoint_path = "training_2/cp.ckpt"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'], optimizer='adam')

    hist = model.fit(X_train, y_train, epochs=10000, batch_size=1,
                     validation_data=(X_test, y_test), callbacks=[cp_callback])