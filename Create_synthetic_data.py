import os
import random

import numpy as np
from PIL import Image

def create_background_from_image_pixels(img_array, number_of_pixels):
    """
    Creates the background of a new image from randomly selected pixels of another image.
    Thus, the background of the created image is similar in color, lighting brightness and
    other parameters to the background of the image from which the pixels were taken.
    Args:
        img_array (np.array): 2D array representing a sample image from which pixels are taken.
        number_of_pixels (int): The number of pixels randomly selected from the image.

    Returns:
        np.array: 2D-array representing the background of the image.
    """
    # Перепроверить аргумент number_of_pixels. Это не буквально количество пикселей, хотя аргумент и связан с их количеством
    # Надо подумать, как его переименовать и описать.
    rows, cols = img_array.shape[0], img_array.shape[1]
    flatten_img_array = img_array.reshape(rows * cols)
    pixels_for_background = np.random.choice(
        flatten_img_array,
        (rows//number_of_pixels) * (cols//number_of_pixels)
    )
    pixels_for_background[pixels_for_background > 0.3] = 0.3
    background_array = pixels_for_background.reshape(
        (rows // number_of_pixels,
         cols // number_of_pixels)
    )
    background_array = np.resize(background_array, (rows, cols))
    return background_array

def preprocess_image(img, height=None, width=None):
    """
    Performs image preprocessing. Converts it to grayscale, resizes it (optional), and normalizes it.
    Args:
        img (PIL.Image): An image that requires preprocessing.
        height (int): The height of the preprocessed image. If None is specified, then image resizing is not performed.
        width (int): The width of the preprocessed image. If None is specified, then image resizing is not performed.

    Returns:
        np.array: A preprocessed 2D array representing an image.
    """
    grayscale_img = img.convert('L')
    if (height != None) or (width != None):
        grayscale_img = grayscale_img.resize((height, width))
    float_img_array = np.asarray(grayscale_img).astype('float32')
    normalized_img_array = float_img_array / 255.0
    return normalized_img_array

def load_preprocessed_image(filepath, height=None, width=None):
    """
    Performs image loading and preprocessing (conversion to grayscale, resizing, normalization)
    Args:
        filepath (str): The path to the image to upload.
        height (int): The height of the preprocessed image. If None is specified, then image resizing is not performed.
        width (int): The width of the preprocessed image. If None is specified, then image resizing is not performed.

    Returns:
        np.array: A preprocessed 2D array representing an image.
    """
    img = Image.open(filepath)
    img_array = preprocess_image(img, height, width)
    return img_array

def names_of_photos(img_path, mask_path):
    """
    Finds the names of all elements previously cut from the training sample photo, as well as their masks.
    Args:
        img_path (str): The path to the elements cut from the training photos.
        mask_path (str): The path to the masks of the elements cut from the training photos.

    Returns:
        tuple: a tuple with two lists. The first contains the names of the elements
               cut from the training photos, the second contains the names of their masks.
    """
    images_paths = [img_path + '/' + name for name in os.listdir(img_path)]
    masks_paths = [mask_path + '/' + name for name in os.listdir(mask_path)]
    return images_paths, masks_paths

def load_image_and_mask(img_path, mask_path):
    """
    Loads and preprocesses the image of an element cut earlier from a training photo, as well as its mask.
    Args:
        img_path (str): The path to the image of the element previously cut from the training photo.
        mask_path (str): The path to the mask image of the element previously cut from the training photo.

    Returns:
        tuple: A tuple containing two 2D arrays. The first is an image of an element previously cut
               from a training photo, the second is an image of its mask.
    """
    img = load_preprocessed_image(img_path)
    mask = load_preprocessed_image(mask_path)
    mask[mask > 0.01] = 1
    return img, mask

def create_random_background(filenames, option, height=None, width=None):
    """
    Creates a background for a new image. The background type is selected randomly from the following options:
    1. The value of all pixels is zero (completely black background)
    2. The pixel value is taken from the create_background_from_image_pixels function. A photo is uploaded from
       the training sample, after which one random pixel is taken from this photo, and a background is created from it.
    3. The pixel value is taken from the create_background_from_image_pixels function. Unlike the previous version,
       in this case, three random pixels are used to create the background.
    4. The background is created from Gaussian noise.
    Args:
        filenames (list): List of names of training photos.
        option (int): Background creation option.
        height (int): The height of the image.
        width (int): The width of the image.

    Returns:
        tuple: A tuple containing two arrays. One is the background of the image,
               the other is the background of its mask. Both are represented by 2D np.arrays.
    """
    filename = random.choice(filenames)
    img = load_preprocessed_image('train/' + filename, height, width)
    background_img = np.zeros((img.shape[0], img.shape[1]))
    if option == 0:
        background_img = np.zeros((background_img.shape[0], background_img.shape[1]))
    elif option == 1:
        background_img = create_background_from_image_pixels(img, 1)
    elif option == 2:
        background_img = create_background_from_image_pixels(img, 3)
    elif option == 3:
        gaussian_noise = np.random.normal(0, 5, background_img.shape)
        background_img += gaussian_noise
        background_img[background_img > 255.0] = 255.0
        background_img[background_img < 0.0] = 0.0
        background_img = background_img / 255.0
    background_mask = np.zeros((background_img.shape[0], background_img.shape[1]))
    return background_img, background_mask

def append_elements_to_background(main_img, main_mask, path, number_of_elements=None):
    """
    Adds elements that were previously cut from training photos, as well as their masks
    to the created background of the new image.
    Args:
        main_img (np.array): A 2D array representing the background of the new image.
        main_mask (np.array): A 2D array representing the background of the new mask.
        path (str): The path where folders with images of elements and their masks are located.
        number_of_elements (int): The number of elements added to the background.

    Returns:
        tuple: a tuple containing two 2D np.arrays. The first is a generated image, the second is its mask.
    """
    if number_of_elements is None:
        number_of_elements = random.randint(30, 60)
    imgs_paths, masks_paths = names_of_photos(path + '/Rocks', path + '/Masks_of_Rocks')
    for counter in range(number_of_elements):
        img_path, mask_path = random.choice(tuple(zip(imgs_paths, masks_paths)))
        current_img, current_mask = load_image_and_mask(img_path, mask_path)
        main_img_insert_row = random.randint(0, main_img.shape[0] - 1)
        main_img_insert_col = random.randint(0, main_img.shape[1] - 1)
        for i in range(current_img.shape[0]):
            if (main_img_insert_row + i >= main_img.shape[0] - 1):
                break
            for j in range(current_img.shape[1]):
                if (main_img_insert_col + j >= main_img.shape[1] - 1):
                    break
                if current_mask[i, j] == 1.0:
                    main_img[main_img_insert_row + i, main_img_insert_col + j] = current_img[i, j]
                main_mask[main_img_insert_row + i, main_img_insert_col + j] = current_mask[i, j]
    main_mask[main_mask != 0] = 1
    return main_img, main_mask

def create_noise_on_image(main_img):
    """
    Adds Gaussian noise to the image.
    Args:
        main_img (np.array): A 2D array representing the generated image.

    Returns:
        np.array: A 2D array representing a generated image with Gaussian noise added to it.
    """
    gaussian_noise = np.random.normal(0, 10, main_img.shape) / 255.0
    main_img += gaussian_noise
    main_img[main_img > 1] = 1
    main_img[main_img > 0] = 0
    return main_img

def save_photo(name, path, im, msk):
    """
    Saves the generated image and its mask at the specified path and under the specified name.
    Args:
        name (str): The name of the image and its mask.
        path (str): The path where the folders for the generated images and their masks are located.
        im (PIL.Image): The generated image.
        msk (PIL.Image): The mask of the generated image.

    Returns:
        None.
    """
    filenames = os.listdir(path)
    if 'Photos_with_rocks' not in filenames:
        os.mkdir(path + '/Photos_with_rocks')
    if 'Photos_with_masks' not in filenames:
        os.mkdir(path + '/Photos_with_masks')
    im.save(path + '/Photos_with_rocks/' + name + '.png')
    msk.save(path + '/Photos_with_masks/' + name + '.png')

if __name__ == '__main__':
    NUMBER_OF_PHOTOS = 200
    filenames = os.listdir('train')
    for name in range(NUMBER_OF_PHOTOS):
        option = random.randint(0, 5)
        background_img, background_mask = create_random_background(filenames, option, 600, 600)
        main_img, main_mask = append_elements_to_background(background_img, background_mask, 'Вырезанные_фото_камней')
        if option == 4:
            main_img = create_noise_on_image(main_img)
        im = Image.fromarray((main_img * 255).astype(np.uint8))
        msk = Image.fromarray((main_mask * 255).astype(np.uint8), 'L')
        if option == 5:
            im.putalpha(msk)
        save_photo(str(name), 'Вырезанные_фото_камней', im, msk)
