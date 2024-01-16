import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

DATADIR = '../Datasets/kagglecatsanddogs_5340/PetImages/'
CATEGORIES = ["Dog", "Cat"]
IMAGE_SIZE = 64
DATASET_SIZE = 4000
TRAIN_SPLIT = 0.9
BATCH_SIZE = 64

def pre_process_image(img_name, path):
    img_array_bgr = cv2.imread(os.path.join(path, img_name))
    img_array_rgb = cv2.cvtColor(img_array_bgr, cv2.COLOR_BGR2RGB)
    img_cropped = cv2.resize(img_array_rgb, (IMAGE_SIZE, IMAGE_SIZE))

    return img_cropped

def flatten_images(train_x, test_x):
    train_x_flatten = train_x.reshape((
    train_x.shape[0],
    -1
    )).T

    test_x_flatten = test_x.reshape((
        test_x.shape[0],
        -1
    )).T

    print(train_x_flatten.shape)
    print(test_x_flatten.shape)

    return train_x_flatten, test_x_flatten

def normalize(train_x, test_x):
    train_x_normalized = train_x / 255.
    test_x_normalized = test_x / 255.

    return train_x_normalized, test_x_normalized

def load_images():
    print('Loading and pre-processing images...')
    images_data = []

    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        image_count = 0

        for img_name in os.listdir(path):
            if image_count != (DATASET_SIZE / 2):
                try:
                    image = pre_process_image(img_name, path)
                    images_data.append([image, class_num])
                    image_count += 1
                except Exception as e:
                    # In case some broken images are found
                    print('Corrupt Image found!')
                    pass
            else:
                break
    
    return images_data

def train_test_split(images):
    print("Generating train and test data...")
    # Spliting Cats and Dogs
    half_split = len(images) // 2
    dogs = images[:half_split]
    cats = images[half_split:len(images)]

    train_split_num = int(len(dogs) * TRAIN_SPLIT)

    train_dogs = dogs[:train_split_num]
    test_dogs = dogs[train_split_num:len(dogs)]
    train_cats = cats[:train_split_num]
    test_cats = cats[train_split_num:len(cats)]

    train_cats_dogs = train_dogs + train_cats
    test_cats_dogs = test_dogs + test_cats

    random.shuffle(train_cats_dogs)
    random.shuffle(test_cats_dogs)

    return train_cats_dogs, test_cats_dogs

def split_features_labels(train_data, test_data):

    X_train = []
    X_test = []
    Y_train = []
    Y_test = []

    for features, labels in train_data:
        X_train.append(features)
        Y_train.append(labels)

    for features1, labels1 in test_data:
        X_test.append(features1)
        Y_test.append(labels1)

    return X_train, X_test, Y_train, Y_test

def generate_dataset(X_train, X_test, Y_train, Y_test):
    print("Generating the dataset...")
    train_x = np.array(X_train)
    test_x = np.array(X_test)
    train_y = np.array(Y_train)
    test_y = np.array(Y_test)

    train_y = train_y.reshape((1, len(train_y)))
    test_y = test_y.reshape((1, len(test_y)))

    print('Shape of Training-X set: ', train_x.shape)
    print('Shape of Testing-X set: ', test_x.shape)
    print('Shape of Training-y set: ', train_y.shape)
    print('Shape of Testing-y set: ', test_y.shape)

    # Flattening the images data
    train_x, test_x = flatten_images(train_x, test_x)
    # Normalizing the images data
    train_x, test_x = normalize(train_x, test_x)

    return train_x, test_x, train_y, test_y

images = load_images()
train_cats_dogs, test_cats_dogs = train_test_split(images)
X_train, X_test, Y_train, Y_test = split_features_labels(train_cats_dogs, test_cats_dogs)
train_x, test_x, train_y, test_y = generate_dataset(X_train, X_test, Y_train, Y_test)
print('\n------- Data loading finished! -------\n\n')
