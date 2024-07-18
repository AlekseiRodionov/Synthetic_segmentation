# Synthetic_segmentation

Здесь представлен код, который автоматизирует процесс разметки фото, полученных с электронного микроскопа. По-сути, этот код представляет собой программу, предназначенную для решения задачи сегментации на фотографиях, полученных с электронного микроскопа. По-сути, это реализация алгоритма, который осуществляет семантическую сегментацию без масок, на которых указаны объекты. Есть только сами фото, без разметки.

Для того, чтобы код работал, необходимо иметь python 3.7, а также следующие библиотеки и модули:
- Numpy
- PIL
- TensorFlow
- Sci-kit learn
- PyQT5

см. файл requirements.txt

Основная программа запускается из файла PyQT_interface.py. При запуске появляется следующее окно: ![alt text]([https://yandex.ru/images/search?from=tabbar&img_url=https%3A%2F%2Fcdn3.iconfinder.com%2Fdata%2Ficons%2Fillustricon-tech%2F512%2Fdevelopment.desktop-1024.png&lr=4&pos=3&rpt=simage&text=картинки%20программа])

Давайте подробно разберём функционал программы.

В первое поле ввода записывается путь к папке с тренировочными файлами. Его же можно выбрать, использовав кнопку справа от поля ввода. Тренировочные данные представляют собой фото с электронного микроскопа, на которых находятся какие-то объекты, которые нужно отделить от фона. Примеры таких фото можно увидеть ниже:

Соответственно, папка с тренировочными данными может выглядеть так:

Во второе поле ввода записывается путь к рабочей директории. В ней будут сохраняться промежуточные результаты, необходимые для дальнейшего обучения модели. Если точнее, в этой папке будут храниться отдельные фото объектов и их маски, созданные при автоматической сегментации, а также синтетические данные — сгенерированные фото и маски к ним.

Путь к рабочей директории также можно ввести самостоятельно или выбрать, нажав на кнопку справа от поля ввода.

Ниже находится кнопка "Автоматическая сегментация". При нажатии на неё выполняется код из файла Automatic_segmentation.py. Алгоритм находит объекты на фото из тренировочной выборки и рисует для них маски. Важно заметить: данный алгоритм ищет границы объектов на основе разности яркостей пикселей, поэтому результат его работы далёк от идеального. Многие объекты не распознаются на фото, а многие распознаются не полностью. Примеры таких дефектных результатов можно увидеть на фото ниже:

Тем не менее, предполагается, что эти результаты подойдут для обучения модели, и она научится хорошо распознавать настоящие объекты на тестовых фото.

После автоматической сегментации в рабочей директории создаются папки "Rocks" и "Masks_of_Rocks". В первую сохраняются изображения объектов, вырезанных с тренировочного фото, а во вторую сохраняются их маски. Также изображения различным образом изменяются (меняется яркость, размер и т.п.), чтобы увеличить разнообразие выборки.


Справа от кнопки "Автоматическая сегментация" находится кнопка "Создание синтетических данных". При нажатии на неё выполняется код из файла Create_synthetic_data.py. Алгоритм создаёт различные фоны, а затем накладывает на них ранее сохранённые изображения объектов, вырезанных с тренировочных фото. Получаются синтетические тренировочные фото. Параллельно для них создаются маски. Примеры созданных синтетических данных можно увидеть ниже:

После создания синтетических данных в рабочей директории создаются папки "Photos_with_rocks", а также "Photos_with_masks". В эти папки сохраняются соответствующие данные.

В третье поле ввода записывается путь к папке, в которую будет сохраняться, а также откуда будет загружаться обученная модель. Этот путь можно ввести вручную или выбрать при нажатии кнопки, находящейся справа от поля ввода.

В четвёртое поле ввода записывается количество эпох, в течение которого модель будет обучаться. Ожидается целое число. Модель сохраняется после каждой эпохи, так что в теории можно прервать обучение, закрыв программу — промежуточные результаты останутся.

Справа от поля ввода количества эпох находится кнопка "Начать обучение". Как следует из названия, при нажатии на неё, модель начинает обучаться (выполняется код из файла Synthetic_fitting.py). Для обучения используются синтетические данные из папок "Photos_with_rocks" и "Photos_with_masks".

В пятое поле ввода записывается путь к тестовому фото, на котором уже обученная модель выполняет семантическую сегментацию. Путь можно ввести вручную или выбрать, нажав на кнопку справа от поля ввода.

Ниже находится кнопка "Выполнить сегментацию". При нажатии на неё выполняется код из файла ML_evaluation.py. По пути, указанному в третьем поле ввода, загружается ранее обученная модель, которая выполняет сегментацию на тестовом фото, путь к которому указан в пятом поле ввода. Результат выполнения сегментации отображается в самой программе:

По окончании сегментации появляется MessageBox со следующей информацией:

Дело в том, что модель часто на синтетических данных обучается находить контуры объектов, но не закрашивает их внутри контуров. Поэтому от пользователя требуется указать фон, чтобы все замкнутые контуры закрасились. Результат закраски контуров показан ниже:

По окончании процесса появляется новый MessageBox со следующей информацией:

Теперь пользователь может выбрать те объекты, которые хорошо были отделены от фона. Для этого достаточно нажать левой кнопкой мыши на нужные объекты. Результат показан ниже:

После выбора первого объекта, под фото появится кнопка "Сохранить выбранные объекты". При нажатии на неё, изображения выбранных объектов вырезаются с фото и сохраняются также, как ранее сохранялись объекты при автоматической сегментации. Рекомендуется для новых объектов использовать другую рабочую директорию. В будущем уже из этих изображений можно создать синтетические данные, на которых будет обучена новая модель. Процесс можно повторять неограниченное количество раз.

В самом низу окна программы находится ProgressBar. Он должен работать при любых вычислениях. В данный момент программа находится в разработке, и алгоритмы не вынесены в отдельные потоки, а выполняются наравне с отрисовкой виджетов. Из-за этого ProgressBar (и всё окно программы) может подвисать во время вычислений.
