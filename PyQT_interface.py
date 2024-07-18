import os
import random
import time

from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap

import Automatic_segmentation
import Create_synthetic_data
import Synthetic_fitting
import ML_evaluation


# The class presented below implements a program for segmenting objects in images from an electron microscope.
# It is assumed that it will not have child classes — this is just a convenient implementation
# of the final version of the project. Most of his methods are to customize the appearance of the program,
# as well as the logic of the widgets. The whole algorithmic part is in the files Automatic_segmentation.py,
# Create_synthetic_data.py, Synthetic_fitting.py, Model_evaluation.py, and to understand the algorithm
# of the program, you can refer to them. The code is documented in detail in them.
# Due to the obvious nature of the code in this file, there will be minimum comments in it.
class Synthetic_segmentation_app():
    """A class for automatic segmentation of objects in images from an electron microscope."""
    def __init__(self):
        self.app = QApplication([])
        self.app.setStyle('Fusion')
        self.window = QWidget()
        self.main_layout = QVBoxLayout()
        self.init_horizontal_layouts(8)
        self.init_line_edits()
        self.init_buttons()
        self.init_text_labels()
        self.init_image_labels()
        self.add_widgets_to_layouts()
        self.connect_buttons()
        self.append_to_main_layout()
        self.window.setLayout(self.main_layout)
        self.window.show()
        self.app.exec_()

    def init_horizontal_layouts(self, n):
        """Creates n horizontal layouts that will be placed on the main layout."""
        self.horizontal_layouts = []
        for i in range(n):
            self.horizontal_layouts.append(QHBoxLayout())

    def init_line_edits(self):
        """Creates all the line_edits that are used in the program."""
        self.file_manager_train_folder_line_edit = QLineEdit()
        self.file_manager_work_folder_line_edit = QLineEdit()
        self.epoch_line_edit = QLineEdit()
        self.file_manager_checkpoint_folder_line_edit = QLineEdit()
        self.file_manager_test_file_line_edit = QLineEdit()

    def init_text_labels(self):
        """Creates all text_labels that are used in the program."""
        self.train_line_text_label = QLabel("Укажите путь к папке с тренировочными фото:")
        self.work_line_text_label = QLabel("Укажите путь к рабочей папке, "
                                            "в которую будут сохраняться синтетические фото:")
        self.epoch_text_label = QLabel("Количество эпох:")
        self.checkpoint_line_text_label = QLabel("Укажите путь к папке, "
                                                 "в которой должна находиться сохранённая модель:")
        self.test_line_text_label = QLabel("Укажите путь к тестовому фото:")

    def init_image_labels(self):
        """Creates labels that are used to display photos in the program."""
        self.test_photo_pixmap = QPixmap().scaled(600, 600)
        self.test_photo_label = QLabel()
        self.test_photo_label.setPixmap(self.test_photo_pixmap)
        self.pred_photo_pixmap = QPixmap().scaled(600, 600)
        self.pred_photo_label = QLabel()
        self.pred_photo_label.setPixmap(self.pred_photo_pixmap)

    def init_buttons(self):
        """Creates all the buttons that will be used in the program."""
        self.evaluate_model_button = QPushButton('Выполнить сегментацию')
        self.file_manager_train_folder_button = QPushButton('...')
        self.file_manager_work_folder_button = QPushButton('...')
        self.file_manager_checkpoint_folder_button = QPushButton('...')
        self.file_manager_test_file_button = QPushButton('...')
        self.automatic_segmentation_button = QPushButton('Автоматическая сегментация')
        self.create_synthetic_data_button = QPushButton('Создание синтетических данных')
        self.fit_model_button = QPushButton('Начать обучение')
        self.save_selected_rocks_button = QPushButton('Сохранить выбранные камни')
        self.save_selected_rocks_button.hide()

    def add_widgets_to_layouts(self):
        """Adds previously created widgets to their respective layouts."""
        self.horizontal_layouts[0].addWidget(self.file_manager_train_folder_line_edit)
        self.horizontal_layouts[0].addWidget(self.file_manager_train_folder_button)
        self.horizontal_layouts[1].addWidget(self.file_manager_work_folder_line_edit)
        self.horizontal_layouts[1].addWidget(self.file_manager_work_folder_button)
        self.horizontal_layouts[2].addWidget(self.automatic_segmentation_button)
        self.horizontal_layouts[2].addWidget(self.create_synthetic_data_button)
        self.horizontal_layouts[3].addWidget(self.file_manager_checkpoint_folder_line_edit)
        self.horizontal_layouts[3].addWidget(self.file_manager_checkpoint_folder_button)
        self.horizontal_layouts[4].addWidget(self.epoch_text_label)
        self.horizontal_layouts[4].addWidget(self.epoch_line_edit)
        self.horizontal_layouts[4].addWidget(self.fit_model_button)
        self.horizontal_layouts[5].addWidget(self.file_manager_test_file_line_edit)
        self.horizontal_layouts[5].addWidget(self.file_manager_test_file_button)
        self.horizontal_layouts[6].addWidget(self.evaluate_model_button)
        self.horizontal_layouts[7].addWidget(self.test_photo_label)
        self.horizontal_layouts[7].addWidget(self.pred_photo_label)

    def append_to_main_layout(self):
        """Adds all previously created layouts, as well as some widgets to the main layout."""
        self.main_layout.addWidget(self.train_line_text_label)
        self.main_layout.addLayout(self.horizontal_layouts[0])
        self.main_layout.addWidget(self.work_line_text_label)
        self.main_layout.addLayout(self.horizontal_layouts[1])
        self.main_layout.addLayout(self.horizontal_layouts[2])
        self.main_layout.addWidget(self.checkpoint_line_text_label)
        self.main_layout.addLayout(self.horizontal_layouts[3])
        self.main_layout.addLayout(self.horizontal_layouts[4])
        self.main_layout.addWidget(self.test_line_text_label)
        self.main_layout.addLayout(self.horizontal_layouts[5])
        self.main_layout.addLayout(self.horizontal_layouts[6])
        self.main_layout.addLayout(self.horizontal_layouts[7])
        for layout in self.horizontal_layouts:
            self.main_layout.addLayout(layout)
        self.main_layout.addWidget(self.save_selected_rocks_button)
        self.progress_bar = QProgressBar()
        self.main_layout.addWidget(self.progress_bar)

    def connect_buttons(self):
        """Connects buttons to methods that are called when they are clicked."""
        self.file_manager_train_folder_button.clicked.connect(self.on_train_folder_button_clicked)
        self.file_manager_work_folder_button.clicked.connect(self.on_work_folder_button_clicked)
        self.file_manager_test_file_button.clicked.connect(self.on_test_file_button_clicked)
        self.automatic_segmentation_button.clicked.connect(self.on_automatic_segmentation_button_clicked)
        self.create_synthetic_data_button.clicked.connect(self.on_create_synthetic_data_button_clicked)
        self.evaluate_model_button.clicked.connect(self.on_evaluate_model_button_clicked)
        self.file_manager_checkpoint_folder_button.clicked.connect(self.on_checkpoint_folder_button_clicked)
        self.fit_model_button.clicked.connect(self.on_fit_model_button_clicked)
        self.save_selected_rocks_button.clicked.connect(self.on_save_selected_rocks_button_clicked)

    def on_train_folder_button_clicked(self):
        """
        Clicking on the button opens the FileDialog to select a directory with training data.
        Then the path to this directory will be inserted into the corresponding lineEdit.
        """
        directory_name = QFileDialog.getExistingDirectory()
        self.file_manager_train_folder_line_edit.setText(directory_name)
        self.file_manager_train_folder_line_edit.clear()
        self.file_manager_train_folder_line_edit.insert(directory_name)

    def on_work_folder_button_clicked(self):
        """
        Clicking on the button opens the FileDialog to select the working directory where the synthetic data
        will be generated. Then the path to this directory will be inserted into the corresponding lineEdit.
        """
        directory_name = QFileDialog.getExistingDirectory()
        self.file_manager_work_folder_line_edit.setText(directory_name)
        self.file_manager_work_folder_line_edit.clear()
        self.file_manager_work_folder_line_edit.insert(directory_name)
    
    def on_checkpoint_folder_button_clicked(self):
        """
        Clicking on the button opens the FileDialog to select the directory where you want to save
        the checkpoint after training the model, or from where you need to download this checkpoint.
        Then the path to this directory will be inserted into the corresponding lineEdit.
        """
        directory_name = QFileDialog.getExistingDirectory()
        self.file_manager_checkpoint_folder_line_edit.setText(directory_name)
        self.file_manager_checkpoint_folder_line_edit.clear()
        self.file_manager_checkpoint_folder_line_edit.insert(directory_name)

    def on_test_file_button_clicked(self):
        """
        Clicking on the button opens the FileDialog to select the test file on which the model
        will perform segmentation. Then the path to this directory will be inserted into the corresponding lineEdit.
        """
        filename = str(QFileDialog.getOpenFileName()[0])
        self.file_manager_test_file_line_edit.setText(filename)
        self.file_manager_test_file_line_edit.clear()
        self.file_manager_test_file_line_edit.insert(filename)

    def train_folder_line_edit_check(self, train_path):
        """
        Checks train_path for correctness. If the path is entered correctly, it returns True.
        If not, it outputs a MessageBox with a description of the problem.
        """
        if train_path.strip() == "":
            error_message = QMessageBox(QMessageBox.Critical,
                                        'Ошибка',
                                        "Задайте путь к папке с тренировочными данными")
            error_message.setStandardButtons(QMessageBox.Ok)
            error_message.exec_()
            return False
        else:
            filenames = os.listdir(train_path)
            flag = False
            for filename in filenames:
                if ('.png' in filename) or ('.jpg' in filename):
                    flag = True
            if not flag:
                error_message = QMessageBox(QMessageBox.Critical,
                                            'Ошибка',
                                            'По указанному пути не найдено ".png" или ".jpg" файлов. '
                                            'Задайте путь к папке с тренировочными данными')
                error_message.setStandardButtons(QMessageBox.Ok)
                error_message.exec_()
                return False
            if flag:
                return True

    def work_folder_line_edit_check(self, work_path):
        """
        Checks the work_path for correctness. If the path is entered correctly, it returns True.
        If not, it outputs a MessageBox with a description of the problem.
        """
        if work_path.strip() == "":
            error_message = QMessageBox(QMessageBox.Critical,
                                        'Ошибка',
                                        "Задайте путь к папке, в которую будут сохраняться синтетические данные")
            error_message.setStandardButtons(QMessageBox.Ok)
            error_message.exec_()
            return False
        else:
            return True

    def checkpoint_folder_line_edit_check(self, checkpoint_path):
        """
        Checks the checkpoint_path for correctness. If the path is entered correctly, it returns True.
        If not, it outputs a MessageBox with a description of the problem.
        """
        if checkpoint_path.strip() == "":
            error_message = QMessageBox(QMessageBox.Critical,
                                        'Ошибка',
                                        "Задайте путь к папке в которой должна быть сохранённая модель. ")
            error_message.setStandardButtons(QMessageBox.Ok)
            error_message.exec_()
            return False
        else:
            return True

    def epoch_line_edit_check(self, epochs):
        """
        Checks the epoch value for correctness. If the path is entered correctly, it returns True.
        If not, it outputs a MessageBox with a description of the problem.
        """
        if epochs.strip() == "":
            error_message = QMessageBox(QMessageBox.Critical,
                                        'Ошибка',
                                        "Введите количество эпох. ")
            error_message.setStandardButtons(QMessageBox.Ok)
            error_message.exec_()
            return False

        else:
            try:
                epochs = int(epochs)
                return True
            except ValueError:
                error_message = QMessageBox(QMessageBox.Critical,
                                            'Ошибка',
                                            'Количество эпох должно быть указано в виде целого числа. '
                                            'Введите количество эпох без посторонних символов.')
                error_message.setStandardButtons(QMessageBox.Ok)
                error_message.exec_()
                return False

    def test_file_line_edit_check(self, file_path):
        """
        Checks the test_path for correctness. If the path is entered correctly, it returns True.
        If not, it outputs a MessageBox with a description of the problem.
        """
        if file_path.strip() == "":
            error_message = QMessageBox(QMessageBox.Critical,
                                        'Ошибка',
                                        "Задайте путь к тестовому фото. ")
            error_message.setStandardButtons(QMessageBox.Ok)
            error_message.exec_()
            return False
        elif ('.png' in file_path) or ('.jpg' in file_path):
            return True
        else:
            error_message = QMessageBox(QMessageBox.Critical,
                                        'Ошибка',
                                        'Тестовое фото должно быть с расширением ".png" или ".jpg". '
                                        'Укажите другое тестовое фото. ')
            error_message.setStandardButtons(QMessageBox.Ok)
            error_message.exec_()
            return False

    def on_automatic_segmentation_button_clicked(self):
        """
        For all photos in the training sample, a simple pixel brightness algorithm determines the contours
        of objects and the objects themselves. This definition occurs with a lot of errors,
        but that's the idea — these errors do not affect the quality of the trained model.
        For more information, see the Readme.
        """
        train_path = self.file_manager_train_folder_line_edit.text()
        work_path = self.file_manager_work_folder_line_edit.text()
        if self.train_folder_line_edit_check(train_path) and self.work_folder_line_edit_check(work_path):
            filenames = os.listdir(train_path)
            for i, filename in enumerate(filenames):
                self.progress_bar.setValue(int(i / len(filenames) * 100))
                img1 = Automatic_segmentation.load_preprocessed_image(train_path + '/' + filename, 600, 600)
                img2 = Automatic_segmentation.extract_rock_borders(img1.copy())
                img2 = Automatic_segmentation.marking_empty_space(img2, 0, 0)
                img2 = Automatic_segmentation.cyclic_marking_empty_space(img2)
                img2 = Automatic_segmentation.cyclic_marking_rocks(img2)
                img2 = Automatic_segmentation.append_borders_to_image(img2)
                img_mask = Automatic_segmentation.create_mask(img2)
                borders_list = Automatic_segmentation.create_rock_borders_list(img2)
                borders_list = Automatic_segmentation.merge_rock_borders(borders_list)
                Automatic_segmentation.save_rocks(str(i), img1, img_mask, borders_list, work_path)
                time.sleep(0.005)
            self.progress_bar.setValue(100)

    def on_create_synthetic_data_button_clicked(self):
        """
        Creates synthetic data from objects located by work_path.
        Please note: the work_path must contain the folders "Masks_of_Rocks" and "Rocks"
        with the corresponding png images.
        """
        train_path = self.file_manager_train_folder_line_edit.text()
        work_path = self.file_manager_work_folder_line_edit.text()
        if self.train_folder_line_edit_check(train_path) and self.work_folder_line_edit_check(work_path):
            NUMBER_OF_PHOTOS = 200
            filenames = os.listdir(train_path)
            for name in range(NUMBER_OF_PHOTOS):
                self.progress_bar.setValue(int(name / NUMBER_OF_PHOTOS * 100))
                option = random.randint(0, 5)
                background_img, background_mask = Create_synthetic_data.create_random_background(filenames, option, 600, 600)
                main_img, main_mask = Create_synthetic_data.append_elements_to_background(background_img, background_mask,
                                                                                          work_path)
                if option == 4:
                    main_img = Create_synthetic_data.create_noise_on_image(main_img)
                im = Image.fromarray((main_img * 255).astype(np.uint8))
                msk = Image.fromarray((main_mask * 255).astype(np.uint8), 'L')
                if option == 5:
                    im.putalpha(msk)
                Create_synthetic_data.save_photo(str(name), work_path, im, msk)
            self.progress_bar.setValue(100)

    def on_fit_model_button_clicked(self):
        """
        Trains the model on data located along the path specified in the work_path. Please note: the folder
        to which the path is specified must contain the folders "Photos_with_masks" and "Photo_with_rocks".
        They must contain the corresponding png images.
        """
        work_path = self.file_manager_work_folder_line_edit.text()
        epochs = self.epoch_line_edit.text()
        self.checkpoint_path = self.file_manager_checkpoint_folder_line_edit.text()
        if self.work_folder_line_edit_check(work_path) and \
                self.checkpoint_folder_line_edit_check(self.checkpoint_path) and \
                self.epoch_line_edit_check(epochs):
            self.checkpoint_path += "/cp.ckpt"
            epochs = int(epochs)
            X_data, y_data = Synthetic_fitting.load_dataset(work_path)
            X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
            model = Synthetic_fitting.Model()

            cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                             save_weights_only=True,
                                                             verbose=1)
            model.compile(loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'], optimizer='adam')
            hist = model.fit(X_train, y_train, epochs=epochs, batch_size=1,
                             validation_data=(X_test, y_test), callbacks=[cp_callback])

    def on_evaluate_model_button_clicked(self):
        """
        The trained model performs semantic segmentation of the image specified by test_path.
        The image is expected in png format.
        """
        test_path = self.file_manager_test_file_line_edit.text()
        self.checkpoint_path = self.file_manager_checkpoint_folder_line_edit.text()
        if self.checkpoint_folder_line_edit_check(self.checkpoint_path) and \
                self.test_file_line_edit_check(test_path):
            self.checkpoint_path += "/cp.ckpt"
            model = ML_evaluation.Model()
            self.progress_bar.setValue(0)
            model.load_weights(self.checkpoint_path)
            self.progress_bar.setValue(10)
            self.img, img_array = ML_evaluation.open_image(test_path)
            self.progress_bar.setValue(20)
            y_pred = model.predict(img_array)
            self.progress_bar.setValue(30)
            y_pred = ML_evaluation.prediction_preprocessing(y_pred)
            self.progress_bar.setValue(40)
            self.mask = Image.fromarray(y_pred)
            self.progress_bar.setValue(50)
            preprocessed_mask = ML_evaluation.mask_preprocessing(self.mask)
            self.progress_bar.setValue(60)
            segmented_img = self.img.copy()
            self.progress_bar.setValue(70)
            segmented_img.paste(preprocessed_mask, (0, 0), preprocessed_mask)
            self.progress_bar.setValue(80)
            segmented_img.save('pred_with_mask.png')
            self.progress_bar.setValue(90)
            self.test_photo_pixmap = QPixmap(test_path).scaled(600, 600)
            self.test_photo_label.setPixmap(self.test_photo_pixmap)
            self.pred_photo_pixmap = QPixmap('pred_with_mask.png').scaled(600, 600)
            self.pred_photo_label.setPixmap(self.pred_photo_pixmap)
            self.pred_photo_label.mousePressEvent = self.painting_closed_contours
            self.progress_bar.setValue(100)
            error_message = QMessageBox(QMessageBox.Information,
                                        'Информация',
                                        "Первичная сегментация выполнена. \n"
                                        "Курсором на правом фото укажите фон (нажмите на пиксель, относящийся к фону), "
                                        "чтобы зарисовать все замкнутые контуры.")
            error_message.setStandardButtons(QMessageBox.Ok)
            error_message.exec_()

    def object_selection(self, event):
        """
        Highlights the selected object mask and the object. In the future, only the selected masks and
        objects will be saved. This is necessary to filter out those objects that were poorly defined by the model.
        """
        self.progress_bar.setValue(0)
        x = event.pos().x()
        y = event.pos().y()
        self.buf_mask = self.buf_mask.convert('RGB')
        rep_value = (255, 255, 0)
        buf_mask = self.buf_mask.load()
        buf_img = self.img.load()
        if buf_mask[x, y] == (255, 255, 255):
            ImageDraw.floodfill(self.buf_mask, (x, y), rep_value, thresh=50)
            self.buf_mask.save('buf_mask.png')
            for i in range(self.img.size[0]):
                for j in range(self.img.size[1]):
                    if buf_mask[i, j] == rep_value:
                        buf_img[i, j] = rep_value
        self.progress_bar.setValue(80)
        segmented_img = self.img.copy()
        segmented_img.paste(self.mask, (0, 0), self.mask)
        self.progress_bar.setValue(80)
        segmented_img.save('pred_with_mask.png')
        self.pred_photo_pixmap = QPixmap('pred_with_mask.png').scaled(600, 600)
        self.pred_photo_label.setPixmap(self.pred_photo_pixmap)
        self.pred_photo_label.mousePressEvent = self.object_selection
        self.save_selected_rocks_button.show()
        self.progress_bar.setValue(100)

    def painting_closed_contours(self, event):
        """
        Defines the background, and then paints over all closed contours that are not related to the background. This
        is necessary because the model often defines the contours of the object itself,
        but does not define the object itself.
        """
        self.progress_bar.setValue(10)
        x = event.pos().x()
        y = event.pos().y()
        self.mask = ML_evaluation.fill_closed_contours(self.mask, (x, y))
        self.progress_bar.setValue(30)
        self.mask = ML_evaluation.mask_preprocessing(self.mask)
        self.progress_bar.setValue(50)
        segmented_img = self.img.copy()
        segmented_img.paste(self.mask, (0, 0), self.mask)
        self.progress_bar.setValue(80)
        segmented_img.save('pred_with_mask.png')
        self.pred_photo_pixmap = QPixmap('pred_with_mask.png').scaled(600, 600)
        self.pred_photo_label.setPixmap(self.pred_photo_pixmap)
        self.pred_photo_label.mousePressEvent = self.object_selection
        self.buf_mask = self.mask.copy()
        self.progress_bar.setValue(100)
        error_message = QMessageBox(QMessageBox.Information,
                                    'Информация',
                                    "Замкнутые контуры закрашены. \n"
                                    "Выберите с помощью курсора на правом фото те объекты, "
                                    "которые были хорошо определены.")
        error_message.setStandardButtons(QMessageBox.Ok)
        error_message.exec_()

    def on_save_selected_rocks_button_clicked(self):
        """
        Saves the objects highlighted in the photo and their masks as separate files
        for further formation of synthetic data.
        """
        work_path = self.file_manager_work_folder_line_edit.text()
        test_path = self.file_manager_test_file_line_edit.text()
        if self.work_folder_line_edit_check(work_path) and self.test_file_line_edit_check(test_path):
            buf_mask = self.buf_mask.load()
            rep_value = (255, 255, 0)
            for i in range(self.img.size[0]):
                for j in range(self.img.size[1]):
                    if buf_mask[i, j] == rep_value:
                        buf_mask[i, j] = (255, 255, 255)
                    else:
                        buf_mask[i, j] = (0, 0, 0)
            self.buf_mask.save(work_path + '/' + 'buf_mask.png')

            self.progress_bar.setValue(int(10))
            img = Automatic_segmentation.load_preprocessed_image(test_path, 600, 600)
            mask = Automatic_segmentation.load_preprocessed_image(work_path + '/' + 'buf_mask.png', 600, 600)
            mask_for_borders = mask.copy()

            mask_for_borders[mask_for_borders == 0.0] = 5.0 # This should be done due to the features of the
            mask_for_borders[mask_for_borders == 1.0] = 3.0 # functions cycling_marking_rocks
                                                            # and append_borders_to_image.
                                                            # These functions will need to be rewritten in the future.

            mask_for_borders = Automatic_segmentation.cyclic_marking_rocks(mask_for_borders)
            mask_for_borders = Automatic_segmentation.append_borders_to_image(mask_for_borders)
            borders_list = Automatic_segmentation.create_rock_borders_list(mask_for_borders)
            borders_list = Automatic_segmentation.merge_rock_borders(borders_list)
            Automatic_segmentation.save_rocks(test_path.split('/')[-1], img, mask, borders_list, work_path)
            time.sleep(0.005)
            self.progress_bar.setValue(100)


my_app = Synthetic_segmentation_app()

