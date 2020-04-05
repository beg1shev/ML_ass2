import matplotlib.pyplot as plt
import csv
import numpy as np
from numpy import pad
from PIL import Image, ImageOps
import random
from skimage import exposure
from skimage import transform
from skimage.util import random_noise
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from cv2 import normalize
from cv2 import NORM_MINMAX
from cv2 import CV_32F
from datetime import datetime

size_x = 20
size_y = 20
n = 43
depth = 10
estimators = 50

def readTrafficSigns(rootpath):
    '''Reads traffic sign data for German Traffic Sign Recognition Benchmark.

    Arguments: path to the traffic sign data, for example './GTSRB/Training'
    Returns:   list of images, list of corresponding labels'''
    images = [] # images
    labels = [] # corresponding labels
    # loop over all 43 classes
    for c in range(0, n):
        prefix = rootpath + '/' + format(c, '05d') + '/' # subdirectory for class
        gtFile = open(prefix + 'GT-'+ format(c, '05d') + '.csv') # annotations file
        gtReader = csv.reader(gtFile, delimiter=';') # csv parser for annotations file
        next(gtReader) # skip header
        # loop over all images in current annotations file
        for row in gtReader:
            images.append(plt.imread(prefix + row[0]))
            labels.append(row[7]) # the 8th column is the label

        gtFile.close()
    return images, labels


def readTest(rootpath):
    test_images = []
    test_labels = []
    gtFile = open(rootpath + '/GT-final_test.csv')
    gtReader = csv.reader(gtFile, delimiter=';')
    next(gtReader)
    for row in gtReader:
        test_images.append(plt.imread(rootpath + "/" + row[0]))
        test_labels.append(row[7])
    gtFile.close()
    return test_images, test_labels


def refactor_images(images):
    for i in range(len(images)):
        if images[i].shape[0] != images[i].shape[1]:
            images[i] = pad_image(images[i])
        images[i] = resize(images[i])
    return images


def resize(image):
    new_im = Image.fromarray(image)
    size = (size_x, size_y)
    resized_image = ImageOps.fit(new_im, size, Image.ANTIALIAS)
    image = np.asarray(resized_image)
    return image


def pad_image(arr):
    if arr.shape[0] < arr.shape[1]:
        if (arr.shape[1] - arr.shape[0]) % 2 == 0:
            padding = arr.shape[1] - arr.shape[0] // 2
            arr = pad(arr, ((padding, padding), (0, 0), (0, 0)), 'constant')
        else:
            padding = (arr.shape[1]- arr.shape[0])
            arr = pad(arr, ((padding, 0), (0, 0), (0, 0)), 'constant')
    elif arr.shape[0] > arr.shape[1]:
        if (arr.shape[0] - arr.shape[1]) % 2 == 0:
            padding = (arr.shape[0] - arr.shape[1]) // 2
            arr = pad(arr, ((0, 0), (padding, padding), (0, 0)), 'constant')
        else:
            padding = arr.shape[0] - arr.shape[1]
            arr = pad(arr, ((0, 0), (padding, 0), (0, 0)), 'constant')
    return arr


def train_validation_split(images, labels):
    train = []
    valid = []

    while images:
        rand = random.random()
        if len(images) < 30:
            if rand < 0.8:
                for i in range(len(images)):
                    train.append((images.pop(), labels.pop()))
            else:
                for i in range(len(images)):
                    train.append((images.pop(), labels.pop()))
        else:
            if rand < 0.8:
                for i in range(30):
                    train.append((images.pop(), labels.pop()))
            else:
                for i in range(30):
                    valid.append((images.pop(), labels.pop()))

    return train, valid


def freq_calc(arr):
    count = [0] * n
    for i in arr:
        count[int(i[1])] += 1
    return count


def augmentation(arr):
    new_arr = arr.copy()
    samples = freq_calc(arr)
    maxx = max(samples)
    length = len(arr) - 1
    num = 0
    for i in range(len(samples)):
        n = samples[i]
        while maxx - n > 0:
            random_num = random.randint(num, num + samples[i])
            for j in range(10):
                new_img = arr[length - random_num][0]
                new_img = augment_image(new_img)
                new_arr.append((new_img, arr[length - random_num][1]))
            n += 10
        num += samples[i]
    return new_arr


def augment_image(im):
    if random.random() > 0.5:
        new_img = random_noise(im)
    else:
        new_img = transform.rotate(im, random.uniform(-20, 20))
    rand = random.random()
    if 0.2 < rand < 0.7:
        new_img = exposure.adjust_gamma(new_img, gamma=rand, gain=rand)
    return new_img


def test_to_matrix(test_arr):
    res = [[0] * (size_x*size_y*3)] * len(test_arr)
    for i in range(len(test_arr)):
        res[i] = test_arr[i].flatten()
    return res


def to_matrix(arr):
    result = [[0] * (size_x*size_y*3)] * len(arr)
    res_labels = []

    for i in range(len(arr)):
        result[i] = arr[i][0].flatten()
        res_labels.append(arr[i][1])
    return result, res_labels


def normalization(matrix):
    result_arr = [[0.0] * (size_x*size_y*3)] * len(matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            value = (float(matrix[i][j]) / 255.0)
            result_arr[i][j] = value
    return result_arr


def train_and_test(feature, label, test_feature, test_label):
    rand_for = RandomForestClassifier(n_estimators=estimators, max_depth=depth)
    rand_for.fit(feature, label)
    prediction = rand_for.predict(test_feature)

    return test_label, prediction


# def plot_chart(plot_arr):
#     scalars = np.arange(0, len(plot_arr))
#     plt.bar(scalars, plot_arr, align='center')
#
#     plt.ylabel("Examples")
#     plt.xlabel("Class")
#     plt.title("Histogram of 43 classes with their number of examples")
#     plt.show()


# def dependency_chart(array1, array2, x_chart, y_chart, title):
#     plt.plot(array1, array2)
#     plt.xlabel(x_chart)
#     plt.ylabel(y_chart)
#     plt.title(title)
#     plt.show()



if __name__ == '__main__':
    # acc_score = [0.56, 0.69, 0.71, 0.71, 0.72]
    # img_size = ["20x20", "25x25", "30x30", "35x35", "40x40"]
    # time_taken = [575, 1077, 1380, 1572, 2853]
    # dependency_chart(img_size, acc_score, "Image size", "Accuracy score", "Dependence of image size on accuracy")
    # dependency_chart(img_size, time_taken, "Image size", "Time(in sec)", "Dependence of image size on time")
    trainImages2, trainLabels = readTrafficSigns('./GTSRB/Final_Training/Images')
    testing_images, testing_labels = readTest('./GTSRB/Final_Test/Images')
    print("Read finished")
    testing_images = refactor_images(testing_images)
    testing_images = test_to_matrix(testing_images)
    trainImages = refactor_images(trainImages2)

    train, validation = train_validation_split(trainImages, trainLabels)
    print("Split finished")
    # before_aug = freq_calc(train)
    # plot_chart(before_aug)
    aug_train = augmentation(train)
    print("Augmentation finished")
    # after_aug = freq_calc(aug_train)
    # plot_chart(after_aug)

    random.shuffle(aug_train)
    random.shuffle(validation)
    random.shuffle(train)

    normal_feature, normal_label = to_matrix(train)
    train_feature, train_label = to_matrix(aug_train)
    valid_feature, valid_label = to_matrix(validation)

    # x0, y0 = train_and_test(normal_feature, normal_label, valid_feature, valid_label)
    # print(classification_report(x0, y0))

    #Augmented data train on validation set
    x1, y1 = train_and_test(train_feature, train_label, valid_feature, valid_label)
    print(classification_report(x1, y1))

    # Non-augmented data train on test set
    x2, y2 = train_and_test(normal_feature, normal_label, testing_images, testing_labels)
    print(classification_report(x2, y2))

    # Augmented data train on test set
    x3, y3 = train_and_test(train_feature, train_label, testing_images, testing_labels)
    print(classification_report(x3, y3))

    normalized_train = normalization(train_feature)
    normalized_test = normalization(testing_images)
    # Augmented, normalized train on test set
    x4, y4 = train_and_test(normalized_train, train_label, normalized_test, testing_labels)
    print(classification_report(x4, y4))




