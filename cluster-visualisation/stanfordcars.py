"""
Reads and parses 'cars_annos' file from the Stanford cars dataset.

The result is an ImageDataset object with the following metadata from the original dataset:
- bounding box values (bbox_x1,bbox_x2,bbox_y1,bbox_y2)
- class name (class)
- file name (fname)
"""
import cv2
import numpy as np
from matplotlib import pyplot
from scipy.ndimage import imread


class StanfordCars(object):

    def __init__(self, annos_dir="cars_annos.csv", image_dir="car_ims/"):
        self.image_dir = image_dir
        # Read and parse meta data
        lines = open(annos_dir).readlines()[1:]
        self.size = len(lines)
        self.bounding_boxes = np.zeros((len(lines), 4))
        self.class_names = np.zeros((len(lines), 1))
        self.file_names = []
        self.__parse_file(lines)

    def read_to_matrix(self, image_shape, classes=None, transform=lambda x: x):
        """
        Creates a matrix of flattened images.
        :param image_shape: tuple (W,H,C): The shape of the resized images that will be put in the matrix.
        :param classes: list: Subset of classes that are included in the image matrix.
        :param transform: function: Transformation to be performed on each image
        :return: array (P, W*H): For P the amount of points corresponding to the classes
        """
        if classes is None:
            classes = self.class_names[:, 0]

        width, height, channels = image_shape
        class_indexes = self.indexes_of_classes(class_indexes=classes)
        matrix = np.zeros((len(class_indexes), width * height * channels))
        for i in range(len(class_indexes)):
            print i
            image = self._read_image(class_indexes[i], (width, height), gray=channels == 1)
            image = transform(image)
            matrix[i, :] = image.flatten()
        return matrix

    def indexes_of_classes(self, class_indexes):
        """
        :param class_indexes: list of indexes of the requested classes
        :return: Array of dataset indexes that have class equal to one of the class_indexes.
        """
        return np.asarray(range(0, self.size))[np.in1d(self.class_names, class_indexes)]

    def cut_bounding_box(self, index):
        """
        :param index: Index of the target image.
        :return: Bounding box cut out of image with index
        """
        image = imread(self.image_dir + self.file_names[index])
        x1, x2, y1, y2 = self.bounding_boxes[index].astype(int)
        return image[y1:y2, x1:x2]

    def _read_image(self, index, shape, gray=False):
        image = cv2.resize(self.cut_bounding_box(index), shape)
        if gray:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def __parse_file(self, lines):
        for i in range(len(lines)):
            line = lines[i].strip().split(",")
            self.bounding_boxes[i, :] = np.asarray([line[0], line[1], line[2], line[3]])
            self.class_names[i, :] = line[4]
            self.file_names.append(line[5])
