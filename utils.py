import matplotlib.pyplot as plt
from tqdm import trange
import os
import numpy as np
import math
from scipy import special
import skimage.measure

#Function to retrive data from files. Return a np array with shape (num_of_steps, min_measurement, max_measurement)

def retrive_dataset(measure_type, start_date, interval):
    base_dir = os.getcwd()
    record_list = []

    for i in range(interval*8):
        filename = base_dir + "/dataset/{}-2021-03-{}-{}.csv".format(measure_type, start_date, i)
        record_list.append(np.loadtxt(filename, delimiter=','))
    
    dataset_raw = np.stack(record_list, axis = 0)
    vmin = np.amin(dataset_raw)
    vmax = np.amax(dataset_raw)
    return dataset_raw, vmin, vmax

#Function to reduce the dimension of the dataset by average pooling.

def average_pooling(dataset, grid_size):
    h, w = grid_size
    kernel_h = math.ceil(dataset.shape[1]/h)
    kernel_w = math.ceil(dataset.shape[2]/w)
    reduced_dataset = np.zeros((dataset.shape[0], h, w))
    for i in range(dataset.shape[0]):
        image = dataset[i, :, :]
        reduced_dataset[i, :, :] = skimage.measure.block_reduce(image, (kernel_h, kernel_w), np.average)
    return reduced_dataset

#Functions to visualize the dataset

def detect_edge(matrix, threshold):
    m, n = matrix.shape
    result = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            if i + 1 < m and abs(matrix[i, j] - matrix[i+1, j]) >= threshold:
                result[i, j] = 1.0
            elif i - 1 >= 0 and abs(matrix[i, j] - matrix[i-1, j]) >= threshold:
                result[i, j] = 1.0
            elif j + 1 < n and abs(matrix[i, j] - matrix[i, j+1]) >= threshold:
                result[i, j] = 1.0
            elif j - 1 >= 0 and abs(matrix[i, j] - matrix[i, j-1]) >= threshold:
                result[i, j] = 1.0
    return result

def visualize_dataset(dataset, measure_type, vmin, vmax, threshold = None):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize = (15, 15))
    i = 0
    for ax in axes.flat:
        image = dataset[i, :, :]
        if threshold is None:
            im = ax.imshow(image, vmin=vmin, vmax=vmax)
        else:
            im = ax.imshow(detect_edge(image, threshold), vmin=0.0, vmax=1.0)
        i+=1

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    title = "{}: the first 9 time steps".format(measure_type)
    if threshold is not None:
        title += ", edge detect with thres={}".format(threshold)
    fig.suptitle(title, fontsize = 24)

    plt.show()