import cupy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

DATADIR = '../Datasets/kagglecatsanddogs_5340/PetImages/'

CATEGORIES = ["Dog", "Cat"]

IMAGE_SIZE = 64
images_data = []
DATASET_SIZE = 5000

def create_dataset():

    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        image_count = 0

        for img_path in os.listdir(path):
            if image_count != (DATASET_SIZE / 2):
                try:
                    img_array_bgr = cv2.imread(os.path.join(path, img_path))
                    img_array_rgb = cv2.cvtColor(
                        img_array_bgr, cv2.COLOR_BGR2RGB)
                    img_cropped = cv2.resize(
                        img_array_rgb, (IMAGE_SIZE, IMAGE_SIZE))
                    images_data.append([img_cropped, class_num])
                    image_count += 1
                except Exception as e:
                    # In case some broken images are found
                    pass
            else:
                break


create_dataset()

half_split = len(images_data) // 2
dogs = images_data[:half_split]
cats = images_data[half_split: len(images_data)]

train_split = int(len(dogs) * 0.8)
test_split = int(len(dogs) * 0.2)

train_dogs = dogs[:train_split]
test_dogs = dogs[train_split: len(dogs)]
train_cats = cats[:train_split]
test_cats = cats[train_split: len(cats)]

train_x_cats_dogs = train_dogs + train_cats
test_x_cats_dogs = test_dogs + test_cats

random.shuffle(train_x_cats_dogs)
random.shuffle(test_x_cats_dogs)

X_train = []
X_test = []
Y_train = []
Y_test = []

for features, labels in train_x_cats_dogs:
    X_train.append(features)
    Y_train.append(labels)

for features1, labels1 in test_x_cats_dogs:
    X_test.append(features1)
    Y_test.append(labels1)

train_raw_x = np.array(X_train)
test_raw_x = np.array(X_test)
train_y = np.array(Y_train)
test_y = np.array(Y_test)

train_y = train_y.reshape((1, len(train_y)))
test_y = test_y.reshape((1, len(test_y)))

print('Shape of Training-X set: ', train_raw_x.shape)
print('Shape of Testing-X set: ', test_raw_x.shape)
print('Shape of Training-y set: ', train_y.shape)
print('Shape of Testing-y set: ', test_y.shape)

train_x_flatten = train_raw_x.reshape((
    train_raw_x.shape[0],
    -1
)).T

test_x_flatten = test_raw_x.reshape((
    test_raw_x.shape[0],
    -1
)).T

print(train_x_flatten.shape)
print(test_x_flatten.shape)

train_x = train_x_flatten / 255.
test_x = test_x_flatten / 255.

print('Data loading finished! \n\n')


# DATADIR = '../Datasets/Car-Bike-Dataset/'
# CATEGORIES = ["Car", "Bike"]

# images_data = []
# # categories_label = []


# def create_image():

#     for categories in CATEGORIES:
#         path = os.path.join(DATADIR, categories)
#         class_num = CATEGORIES.index(categories)
#         image_count = 0

#         for img_path in os.listdir(path):
#             if image_count != 2000:
#                 try:
#                     img_array_bgr = cv2.imread(os.path.join(path, img_path))
#                     img_array_rgb = cv2.cvtColor(
#                         img_array_bgr, cv2.COLOR_BGR2RGB)
#                     img_cropped = cv2.resize(img_array_rgb, (64, 64))
#                     images_data.append([img_cropped, class_num])
# #                     categories_label.append(class_num)
#                     image_count += 1

#                 except:
#                     pass


# create_image()

# train_set = images_data[:1750]
# train_set.extend(images_data[2000:3750])

# test_set = images_data[1750:2000]
# test_set.extend(images_data[3750:])

# random.shuffle(train_set)
# random.shuffle(test_set)


# train_x_set_org = []
# train_y_set = []
# for i in range(3500):
#     train_x_set_org.append(train_set[i][0])
#     train_y_set.append(train_set[i][1])

# test_x_set_org = []
# test_y_set = []
# for i in range(500):
#     test_x_set_org.append(test_set[i][0])
#     test_y_set.append(test_set[i][1])

# train_x_set_org = np.array(train_x_set_org)
# train_y = np.array(train_y_set).reshape(1, 3500)

# test_x_set_org = np.array(test_x_set_org)
# test_y = np.array(test_y_set).reshape(1, 500)

# print('Shape of Training-X set: ', train_x_set_org.shape)
# print('Shape of Testing-X set: ', test_x_set_org.shape)
# print('Shape of Training-y set: ', train_y.shape)
# print('Shape of Testing-y set: ', test_y.shape)

# train_x_set_flatten = train_x_set_org.reshape(train_x_set_org.shape[0], -1).T
# test_x_set_flatten = test_x_set_org.reshape(test_x_set_org.shape[0], -1).T

# train_x = train_x_set_flatten/255.
# test_x = test_x_set_flatten/255.


# print('Data loading finished! \n\n')