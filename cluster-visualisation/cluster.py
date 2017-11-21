"""
Visualise images using TSNE
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from cegeka.stanfordcars import StanfordCars
from cegeka.utils.plot import show_image

WIDTH = 100
HEIGHT = 100
CHANNELS = 3
IMAGE_SHAPE = (WIDTH, HEIGHT, CHANNELS)


def preprocess(data):
    preprocessed = cv2.blur(data, (2, 2))
    return preprocessed


def perform_pca(data, n_components=40):
    pca = PCA(n_components=n_components)
    pca.fit(data)
    return pca.transform(data)


def perform_tsne(data, tsne_object=TSNE(n_components=2, perplexity=50.0, n_iter=1000, verbose=1)):
    return tsne_object.fit_transform(data)


def imscatter(x, y, images, imread=plt.imread, zoom=1.0):
    fig, ax = plt.subplots()
    for x0, y0, image in zip(x, y, images):
        im = OffsetImage(imread(image), zoom=zoom)
        ax.add_artist(AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    ax.scatter(x, y)
    plt.show()


if __name__ == "__main__":
    image_dataset = StanfordCars(annos_dir="devkit/cars_annos.csv",
                                 image_dir="data/")

    classes = [range(101,105) + range(145, 150) + range(39,45)] # Ferrari & Jeep & Bentley

    print("Preprocessing data...")
    data = image_dataset.read_to_matrix(IMAGE_SHAPE,
                                        classes=classes,
                                        transform=preprocess)

    print("Reducing data with PCA...")
    data = perform_pca(data, n_components=40)

    print("Performing TSNE...")
    data = perform_tsne(data, tsne_object=TSNE(n_components=2, perplexity=50.0, n_iter=1000, verbose=1))

    imscatter(data[:, 0], data[:, 1],
              data.astype(list),
              imread=lambda x: np.reshape(x.astype(int), (100, 100, 3))/255.0,
              zoom=0.15)
