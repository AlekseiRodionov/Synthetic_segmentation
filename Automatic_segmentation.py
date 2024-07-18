import random
import os

import numpy as np
from PIL import Image


def preprocess_image(img, height, width):
    """
    Performs basic transformations of the uploaded image
    (grayscale conversion, resizing, type conversion, normalization).
    Args:
        img (np.ndarray): Image, represented as a PIL.Image.
        height (int): The height of the image. Equal to the number of rows.
        width (int): The width of the image. Equal to the number of cols.

    Returns:
        np.ndarray: Preprocessed image, represented as a numpy-array.
    """
    grayscale_img = img.convert('L')
    resized_img = grayscale_img.resize((height, width))
    float_img_array = np.asarray(resized_img).astype('float32')
    normalized_img_array = float_img_array / 255.0
    return normalized_img_array

def fill_space_pixel(img_array, row_index, column_index, recursion_depth=0, max_recursion_depth=None):
    """
    Fills in one pixel if this pixel belongs to an empty space
    (it is no longer filled in and is not the boundary of the stone)
    Args:
        img_array: An image with a borders.
        row_index (int): The row of current pixel.
        column_index (int): The column of current pixel.
        recursion_depth (int): Current recursion level.
        max_recursion_depth (int): The maximum level of recursion. If the image size is too large,
                                   the computer's memory may not be enough to recursively fill
                                   the entire image. Therefore, the recursion depth limit is used.

    Returns:
        None.
    """
    if (img_array[row_index, column_index] != 1) and (img_array[row_index, column_index] != 5):
        img_array[row_index, column_index] = 2
        if (recursion_depth < max_recursion_depth) or (max_recursion_depth is None):
            marking_empty_space(img_array, row_index, column_index, recursion_depth)

def fill_rock_pixel(filled_img_array, row_index, column_index, recursion_depth=0, max_recursion_depth=None):
    """
    Fills in one pixel if this pixel belongs to the rock
    (it is no longer filled in and is not the boundary of the rock)
    Args:
        filled_img_array (np.ndarray): An image with a marked-up empty space.
        row_index (int): The row of current pixel.
        column_index (int): The column of current pixel.
        recursion_depth (int): Current recursion level.
        max_recursion_depth (int): The maximum level of recursion. If the image size is too large,
                                   the computer's memory may not be enough to recursively fill
                                   the entire image. Therefore, the recursion depth limit is used.
    Returns:
        None.

    """
    if (filled_img_array[row_index, column_index] != 5) and (filled_img_array[row_index, column_index] != 3):
        filled_img_array[row_index, column_index] = 2
        if (recursion_depth < max_recursion_depth) or (max_recursion_depth is None):
            marking_rocks(filled_img_array, row_index, column_index, recursion_depth)

def is_valid_y_cell(image_pixels, x, y):
    """
    Checks whether the point with coordinates [x, y], or the point
    to the left or right of it is part of the image of the rock.
    Args:
        image_pixels (np.ndarray): Preprocessed image.
        x (int): The row of the pixel.
        y (int): The column of the pixel.

    Returns:
        bool: A flag indicating whether the cell is valid.
    """
    flag = False
    if y - 1 >= 0:
        if image_pixels[x, y - 1] == 3:
            flag = True
    if image_pixels[x, y] == 3:
        flag = True
    if y + 1 <= image_pixels.shape[1] - 1:
        if image_pixels[x, y + 1] == 3:
            flag = True
    return flag

def is_valid_cell(img_array, x, y):
    """
    Checks whether the point is valid. In order for it to be valid, it is necessary that there is at least
    one point in the neighborhood that makes up the image of the rock (not an empty space), and at the same time
    the current point has not been passed by the algorithm before (did not have the value -1),
    and is also not part of the image of the rock.
    Args:
        image_pixels (np.ndarray): Preprocessed image.
        x (int): The row of the pixel.
        y (int): The column of the pixel.

    Returns:
        bool: A flag indicating whether the cell is valid.
    """
    flag = False
    if x - 1 >= 0:
        flag += is_valid_y_cell(img_array, x - 1, y)
    flag += is_valid_y_cell(img_array, x, y)
    if x + 1 <= img_array.shape[0] - 1:
        flag += is_valid_y_cell(img_array, x + 1, y)
    if flag and (img_array[x, y] != -1) and (img_array[x, y] != 3):
        return True
    else:
        return False

def find_rock_borders(img_array, row_index, column_index):
    """
    Starting from the point with row_index and column_index coordinates, which is part of the image of the stone
    in the general image, searches for the boundaries of this stone and calculates the extreme coordinates
    of these boundaries (left, right, top, bottom). To do this, the algorithm first finds the uppermost point
    of the rock in the image (the lowest in the array), after which it begins to bypass the contour
    of the rock, finding the leftmost, rightmost, uppermost and lowest points in the process.
    Args:
        img_array (np.ndarray): Preprocessed image.
        row_index (int): The row of the image point from which the boundary search begins.
        column_index (int): The column of the image point from which the boundary search begins.

    Returns:
        tuple: The extreme coordinates of the boundaries (left, right, top, bottom).
    """
    left, right, top, bottom = column_index, column_index, row_index, row_index
    i, j = row_index, column_index
    while (i > 0) and (img_array[i, j] == 3):
        i -= 1
        top = i

    while True:
        if (i - 1 >= 0) and is_valid_cell(img_array, i - 1, j):
            if top > i - 1:
                top = i - 1
            i = i - 1
        elif (j - 1 >= 0) and is_valid_cell(img_array, i, j - 1):
            if left > j - 1:
                left = j - 1
            j = j - 1
        elif (i + 1 < img_array.shape[0]) and is_valid_cell(img_array, i + 1, j):
            if bottom < i + 1:
                bottom = i + 1
            i = i + 1
        elif (j + 1 < img_array.shape[1]) and is_valid_cell(img_array, i, j + 1):
            if right < j + 1:
                right = j + 1
            j = j + 1
        else:
            return left, right, top, bottom
        img_array[i, j] = -1

def is_point_in_borders_list(borders_list, i, j):
    """
    Checks whether a point with coordinates i and j falls inside
    at least one rectangle described by the coordinates of the borders
    contained in the borders_list. This is necessary in order to skip the
    pixels that make up the stones, for which the boundaries have
    already been highlighted.
    Args:
        borders_list (list): a list whose elements are the coordinates of the boundaries of the rocks in the image.
        i (int): The row of the image point to be checked.
        j (int): The column of the image point to be checked.

    Returns:
        bool: A flag indicating whether a point is inside the boundaries of one of the rocks, or not
    """
    flag = True
    for element in borders_list:
        left, right = element[1]
        top, bottom = element[0]
        if (top <= i <= bottom) and (left <= j <= right):
            flag = False
            break
    return flag

def save_image_and_mask(name, img_array, mask_array, img_width, img_height, path='Вырезанные_фото_камней'):
    """
    Saves the image of the stone, as well as the mask image with the specified width and height of the image
    along the specified path. The image of the rock is saved to the "Rocks" folder, and the image of the mask
    is saved to the "Masks_of_rocks" folder
    (if such folders do not exist in the specified path, they are created automatically).
    Both the image of the rock and the mask use the same filename.
    Args:
        name (str): The name for the image of the rock and the image of the mask.
        img_array (np.ndarray): Preprocessed image of rock.
        mask_array (np.ndarray): The mask indicating where the rock is and where the empty space is.
        img_width (int): Width of the images.
        img_height (int): Height of the images.
        path (str): The path where the files are saved (the specified path should contain the
                    folders "Rocks" and "Masks_of_Rocks". If there are none, they are created automatically.
                    Files are saved in these folders).

    Returns:
        None
    """
    names_on_specified_path = os.listdir(path)
    if 'Rocks' not in names_on_specified_path:
        os.mkdir(path + '/' + 'Rocks')
    if 'Masks_of_Rocks' not in names_on_specified_path:
        os.mkdir(path + '/' + 'Masks_of_Rocks')
    new_im = Image.fromarray((img_array * 255).astype(np.uint8))
    new_msk = Image.fromarray((mask_array * 255).astype(np.uint8))
    new_im = new_im.resize((img_width, img_height))
    new_msk = new_msk.resize((img_width, img_height))
    new_im.save(path + '/Rocks/' + name + '.png')
    new_msk.save(path + '/Masks_of_Rocks/' + name + '.png')

def change_to_white(name, img_array, mask_array, img_width=600, img_height=600, percent=0.95):
    """
    The mask paints over the stone in the image in white
    (The brightness will be set to 0.8 + a random value from 0 to 0.2).
    At the same time, with a certain chance, the pixel may not be colored.
    The string "_white_mask" is added to the file name.
    Args:
        name (str): The name for the image of the rock and the image of the mask.
        img_array (np.ndarray): Preprocessed image of rock.
        mask_array (np.ndarray): The mask indicating where the rock is and where the empty space is.
        img_width (int): Width of the images.
        img_height (int): Height of the images.
        percent (float): The chance that the pixel will be colored white.

    Returns:
        None
    """
    for i in range(mask_array.shape[0]):
        for j in range(mask_array.shape[1]):
            if (mask_array[i, j] != 0) and (random.random() <= percent):
                img_array[i, j] = 0.8 + random.random() / 5
    name = name + '_white_mask'
    save_image_and_mask(name, img_array, mask_array, img_width, img_height)

def change_brightness(name, img, mask, x, y):
    """
    Changes the brightness of the image randomly (up to 50%). Adds the string "_brightness" to the file name.
    Args:
        name (str): The name for the image of the rock and the image of the mask.
        img_array (np.ndarray): Preprocessed image of rock.
        mask_array (np.ndarray): The mask indicating where the rock is and where the empty space is.
        img_width (int): Width of the images.
        img_height (int): Height of the images.

    Returns:
        None
    """
    brightness = random.random() / 2
    img = img + brightness
    img[img < 0] = 0
    img[img > 1] = 1
    name = name + '_brightness'
    save_image_and_mask(name, img, mask, x, y)

def change_geometry(name, img_array, mask_array, img_width=600, img_height=600):
    """
    Resizes the image and the image mask (width and height) randomly,
    but so that the width and height cannot be reduced or increased more than three times.
    This restriction is chosen so that images that are too small or too large are not obtained.
    Args:
        i (int): The image number used as part of the name.
        img_array (np.ndarray): Preprocessed image of rock.
        mask_array (np.ndarray): The mask indicating where the rock is and where the empty space is.
        img_width (int): Current width of the images.
        img_height (int): Current height of the images.

    Returns:
        None
    """
    new_img_width = random.randint(img_width // 3 + 3, img_width * 3 + 3)
    new_img_height = random.randint(img_height // 3 + 3, img_height * 3 + 3)
    new_name = name + '_change_geometry'
    save_image_and_mask(new_name, img_array, mask_array, new_img_width, new_img_height)

def load_preprocessed_image(filepath: str, height: int, width: int) -> np.ndarray:
    """
    The function loads the image at the specified path, and also performs its preprocessing.
    Args:
        filepath (str): The path to the uploaded image.
        height (int): The height that the uploaded image should have after preprocessing. Equal to the number of rows.
        width (int): The width that the uploaded image should have after preprocessing. Equal to the number of cols.

    Returns:
        np.ndarray: Preprocessed image, represented as a numpy-array.
    """
    img = Image.open(filepath)
    img_array = preprocess_image(img, height, width)
    return img_array

def extract_rock_borders(img_array: np.ndarray):
    """
    The function highlights the boundaries of the stones in the image.
    In fact, pixels are highlighted whose brightness is above a certain threshold.
    Thus, not all borders are highlighted correctly, but that's the point.
    This is enough for the method used.
    Args:
        img_array (np.ndarray): Image, represented as a numpy-array.

    Returns:
        np.ndarray: An image with highlighted rock borders.
    """
    brightest_pixels = np.where(
        img_array > np.percentile(img_array, 5),
        0,
        img_array)
    brightest_pixels[brightest_pixels != 0] = 1
    return brightest_pixels

def marking_empty_space(extracted_borders_array, row_index, column_index, recursion_depth=0, max_recursion_depth=10):
    """
    The function recursively traverses the entire image, bypassing closed contours,
    and marks an empty space that is not a stone. Since there may not be enough memory
    to recursively traverse the entire image, it is assumed that this function will be
    called in a loop. It is also worth explaining that if the boundaries of the stones
    do not form a closed contour, then the stone will be marked as an empty space.
    Args:
        extracted_borders_array (np.ndarray): An image with highlighted rock borders, represented as a numpy array.
        row_index (int): The row from which the pixel filling begins.
        column_index (int): The column from which the pixel filling begins.
        recursion_depth (int): Current recursion level.
        max_recursion_depth (int): The maximum level of recursion. If the image size is too large,
                                   the computer's memory may not be enough to recursively fill
                                   the entire image. Therefore, the recursion depth limit is used.
    Returns:
        np.ndarray: An image with partially mapped empty space, presented as a numpy array.
    """
    recursion_depth += 1
    if (extracted_borders_array[row_index, column_index] not in (1, 5)):
        extracted_borders_array[row_index, column_index] = 5
        if row_index - 1 >= 0:
            fill_space_pixel(extracted_borders_array, row_index - 1, column_index, recursion_depth, max_recursion_depth)
        if column_index - 1 >= 0:
            fill_space_pixel(extracted_borders_array, row_index, column_index - 1, recursion_depth, max_recursion_depth)
        if row_index + 1 < extracted_borders_array.shape[0]:
            fill_space_pixel(extracted_borders_array, row_index + 1, column_index, recursion_depth, max_recursion_depth)
        if column_index + 1 < extracted_borders_array.shape[1]:
            fill_space_pixel(extracted_borders_array, row_index, column_index + 1, recursion_depth, max_recursion_depth)
    return extracted_borders_array

def cyclic_marking_empty_space(extracted_borders_array):
    """
    The function in the loop calls marking_empty_space (see docstring of marking_empty_space).
    This is necessary in order to get around the problem of running out of memory when recursion is too deep
    Args:
        extracted_borders_array (np.ndarray): An image with highlighted rock borders, represented as a numpy array.

    Returns:
        np.ndarray: An image with a marked empty space, represented as numpy array.
    """
    while True:
        flag = True
        for i in range(extracted_borders_array.shape[0]):
            for j in range(extracted_borders_array.shape[1]):
                if extracted_borders_array[i, j] == 2:
                    extracted_borders_array = marking_empty_space(extracted_borders_array, i, j)
                    flag = False
        if flag:
            break
    return extracted_borders_array

def marking_rocks(filled_img_array, row_index, column_index, recursion_depth=0, max_recursion_depth=10):
    """
    The function recursively traverses the entire image, bypassing closed contours,
    and marks rocks. Since there may not be enough memory
    to recursively traverse the entire image, it is assumed that this function will be
    called in a loop.
    Args:
        filled_img_array (np.ndarray): An image with a marked-up empty space.
        row_index (int): The row from which the pixel filling begins.
        column_index (int): The column from which the pixel filling begins.
        recursion_depth (int): Current recursion level.
        max_recursion_depth (int): The maximum level of recursion. If the image size is too large,
                                   the computer's memory may not be enough to recursively fill
                                   the entire image. Therefore, the recursion depth limit is used.
    Returns:
        np.ndarray: An image with partially mapped rocks, presented as a numpy array.

    """
    recursion_depth += 1
    if filled_img_array[row_index, column_index] != 5:
        filled_img_array[row_index, column_index] = 3
        if row_index - 1 >= 0:
            fill_rock_pixel(filled_img_array, row_index - 1, column_index, recursion_depth, max_recursion_depth)
        if column_index - 1 >= 0:
            fill_rock_pixel(filled_img_array, row_index, column_index - 1, recursion_depth, max_recursion_depth)
        if row_index + 1 < filled_img_array.shape[0]:
            fill_rock_pixel(filled_img_array, row_index + 1, column_index, recursion_depth, max_recursion_depth)
        if column_index + 1 < filled_img_array.shape[1]:
            fill_rock_pixel(filled_img_array, row_index, column_index + 1, recursion_depth, max_recursion_depth)
    return filled_img_array

def cyclic_marking_rocks(filled_img_array):
    """
    The function in the loop calls marking_rocks (see docstring of marking_rocks).
    This is necessary in order to get around the problem of running out of memory when recursion is too deep.
    Args:
        filled_img_array (np.ndarray): An image with a marked-up empty space.

    Returns:
        np.ndarray: An image with a marked-up empty space and rocks.
    """
    while True:
        flag = True
        for i in range(filled_img_array.shape[0]):
            for j in range(filled_img_array.shape[1]):
                if filled_img_array[i, j] in (0, 2):
                    filled_img_array = marking_rocks(filled_img_array, i, j)
                    flag = False
        if flag:
            break
    return filled_img_array

def create_mask(marked_img_array):
    """
    Creating a mask of the original image, which indicates where the stones are and where the empty space is.
    The stones are marked with 1, and the empty space with 0
    Args:
        marked_img_array (np.ndarray): An image on which empty space and rocks are marked separately.

    Returns:
        np.ndarray: The mask of the image. 1 - rock, 0 - empty space.
    """
    img_mask = marked_img_array.copy()
    img_mask[img_mask == 5] = 0
    img_mask[(img_mask == -1) | (img_mask == 3)] = 1
    return img_mask

def append_borders_to_image(marked_img_array):
    """
    Fills in the extreme pixels of the image (the first and last row, the first and last column).
    This is necessary to create closed contours for stones that go beyond the boundaries of the image.
    Args:
        marked_img_array (np.ndarray): An image on which empty space and rocks are marked separately.

    Returns:
        np.ndarray: An image with the first and last row marked up, as well as the first and last column.
    """
    marked_img_array[marked_img_array == 1] = 5
    marked_img_array[0, :] = 5
    marked_img_array[marked_img_array.shape[0] - 1, :] = 5
    marked_img_array[:, 0] = 5
    marked_img_array[:, marked_img_array.shape[1] - 1] = 5
    return marked_img_array

def create_rock_borders_list(img_array):
    """
    Creates a list of boundaries along which rocks will be cut from the image
    in the future to create a synthetic training sample.
    Args:
        img_array (np.ndarray): Preprocessed image.

    Returns:
        list: List of borders.
    """
    borders_list = []
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            if (img_array[i, j] == 3) and (is_point_in_borders_list(borders_list, i, j)):
                left, right, top, bottom = find_rock_borders(img_array, i, j)
                borders_list.append([(top, bottom), (left, right)])
    return borders_list

def merge_rock_borders(borders_list):
    """
    Unites intersecting borders. If the rocks are located close to each other, and they cannot
    be cut from the image separately without capturing part of another rock, the borders are combined
    so that all the stones are cut together, and there are no cut parts.
    Args:
        borders_list (list): The list of borders where the rocks will be cut.

    Returns:
        list: The list of updated borders.
    """
    i = 0
    while i < len(borders_list):
        j = 0
        while j < len(borders_list):
            if i == j:
                j += 1
                continue
            if i >= len(borders_list) or j >= len(borders_list):
                break
            top1, bottom1, left1, right1 = *borders_list[i][0], *borders_list[i][1]
            top2, bottom2, left2, right2 = *borders_list[j][0], *borders_list[j][1]
            if ((top2 <= bottom1) and (top2 >= top1)) and ((left2 <= right1) and (left2 >= left1)):
                top_result = min(top1, top2)
                bottom_result = max(bottom1, bottom2)
                left_result = min(left1, left2)
                right_result = max(right1, right2)
                borders_list[i] = [(top_result, bottom_result), (left_result, right_result)]
                del borders_list[j]
            else:
                j += 1
        i += 1
    return borders_list

def save_rocks(filename, img_array, mask_array, borders_list, path):
    """
    Carves rocks according to the boundaries found, performs various transformations with them
    and saves them with masks as separate images.
    Args:
        img_array (np.ndarray): Preprocessed image.
        mask_array (np.ndarray): An image mask indicating where the rock is and where the empty space is.
        borders_list (list): The list of borders where the stones will be cut.
        path (str): Path to the work directory.

    Returns:
        None.
    """
    i = 0
    for borders in borders_list:
        top, bottom, left, right = *borders[0], *borders[1]
        img_width = abs(left - right)
        img_height = abs(bottom - top)
        name = filename + '_' + str(i)
        new_img = img_array[top:bottom, left:right]
        new_mask = mask_array[top:bottom, left:right]
        save_image_and_mask(name, new_img, new_mask, img_width, img_height, path)
        change_geometry(name, new_img, new_mask, img_width, img_height)
        change_brightness(name, new_img, new_mask, img_width, img_height)
        change_to_white(name, new_img, new_mask, img_width, img_height, percent=1)
        i += 1

if __name__ == '__main__':
    filenames = os.listdir('train')
    for filename in filenames:
        img1 = load_preprocessed_image('train/' + filename, 600, 600)
        img2 = extract_rock_borders(img1.copy())
        img2 = marking_empty_space(img2, 0, 0)
        img2 = cyclic_marking_empty_space(img2)
        img2 = cyclic_marking_rocks(img2)
        img2 = append_borders_to_image(img2)
        img_mask = create_mask(img2)

        borders_list = create_rock_borders_list(img2)
        borders_list = merge_rock_borders(borders_list)
        save_rocks(filename, img1, img_mask, borders_list)