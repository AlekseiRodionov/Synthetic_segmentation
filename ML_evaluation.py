import tensorflow as tf
from PIL import Image, ImageDraw
import numpy as np

CHECKPOINT_PATH = "training_2/cp.ckpt"
TEST_IMAGE_PATH = '21-1_1-Image 0001-0-0.jpg'
FILL_POINT = (100, 0)


def open_image(path):
    """
    Opens the image. Returns both a PIL.Image and a representation of this image as an np.array.
    Args:
        path (str): The path to the image.

    Returns:
        tuple: A tuple containing PIL.Image, as well as its representation in the form of np.array.
    """
    img = Image.open(path).convert('RGB')
    img = img.resize((600, 600))
    img_array = np.array(img)[None, ...] / 255.0
    return img, img_array

def prediction_preprocessing(y_pred):
    """
    Performs preprocessing of the predicted image mask. All fractional values are rounded to integers.
    Then the values of individual pixels are translated from the interval [0, 1] to the interval [0, 255]
    Args:
        y_pred (np.array): An array representing the predicted image mask.
                           The value of each pixel is in the range from 0 to 1.
    Returns:
        np.array: A preprocessed array representing the predicted image mask.
    """
    y_pred[y_pred >= 0.5] = 1.0
    y_pred[y_pred < 0.5] = 0.0
    y_pred = (y_pred[0] * 255.0).astype(np.uint8)[:, :, 0]
    return y_pred

def fill_closed_contours(mask, starting_point):
    """
    Fills all closed contours, making the masks inside solid. The starting point should point to the background.
    Args:
        mask (PIL.Image): Predicted image mask.
        starting_point (tuple): The starting point of the fill. It is a tuple with two coordinates (row and column).
                                The starting point should point to the background.
    Returns:
        PIL.Image: The predicted image mask with filled closed contours.
    """
    mask = mask.convert('RGB')
    rep_value = (255, 255, 0)
    ImageDraw.floodfill(mask, starting_point, rep_value, thresh=50)

    for i in range(600):
        for j in range(600):
            fill_point = (i, j)
            rep_value = (255, 255, 255)
            if mask.getpixel(fill_point) == (0, 0, 0):
                ImageDraw.floodfill(mask, fill_point, rep_value, thresh=50)

    rep_value = (0, 0, 0)
    ImageDraw.floodfill(mask, starting_point, rep_value, thresh=50)
    return mask

def mask_preprocessing(mask):
    """
    Performs preprocessing of the image mask for its further superimposition on the image.
    Args:
        mask (PIL.Image): Predicted image mask.

    Returns:
        PIL.Image: Preprocessed predicted image mask
    """
    mask = mask.convert("RGBA")
    data = mask.getdata()
    newData = []
    for item in data:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    mask.putdata(newData)
    return mask

def show_results(img, mask):
    """
    Shows the results of the neural network operation. Opens the original image from which the prediction was made,
    and then opens a new image on which the entire background is cut out and only the objects for which
    the mask was obtained are left.
    Args:
        img (PIL.Image): The original image.
        mask (PIL.Image): The mask used to highlight objects in the image.

    Returns:
        None.
    """
    img.show()
    img.paste(mask, (0, 0), mask)
    img.show()


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
    model = Model()
    model.load_weights(CHECKPOINT_PATH)
    img, img_array = open_image(TEST_IMAGE_PATH)
    y_pred = model.predict(img_array)
    y_pred = prediction_preprocessing(y_pred)
    mask = Image.fromarray(y_pred)
    mask = fill_closed_contours(mask, FILL_POINT)
    mask = mask_preprocessing(mask)
    show_results(img, mask)