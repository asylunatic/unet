from PIL import Image, ImageSequence
import elasticdeform
import numpy as np
import os
import glob
import time

np.random.seed(1)


def makedir(path):
    current_directory = os.getcwd()
    final_directory = os.path.join(current_directory, rf'{path}')
    if not os.path.exists(final_directory):
        os.makedirs(final_directory)


def crop_center(image, crop_x, crop_y):
    y, x = image.shape
    start_x = x // 2 - (crop_x // 2)
    start_y = y // 2 - (crop_y // 2)
    return image[start_y:start_y + crop_y, start_x:start_x + crop_x]


def augment_image(image, label, output_size=(700, 700)):
    image_array = np.array(image)
    label_array = np.array(label)

    # ensure there is enough padding in case output width = image width
    pad_width = output_size[0] - image_array.shape[0] + 200
    pad_height = output_size[1] - image_array.shape[1] + 200

    image_array = np.pad(image_array, pad_width=[(pad_width, pad_width), (pad_height, pad_height)], mode='symmetric')
    label_array = np.pad(label_array, pad_width=[(pad_width, pad_width), (pad_height, pad_height)], mode='symmetric')

    image_array, label_array = elasticdeform.deform_random_grid([image_array, label_array], sigma=10, points=3,
                                                                order=[3, 0])

    image_array = crop_center(image_array, output_size[0], output_size[1])
    label_array = crop_center(label_array, output_size[0], output_size[1])

    return image_array, label_array


def augment_segmentation_dataset(factor):
    train_output_path = f'segmentation_challenge_data/augmented/train_{factor}x'
    label_output_path = f'segmentation_challenge_data/augmented/labels_{factor}x'
    makedir(train_output_path)
    makedir(label_output_path)

    training_image_volume = Image.open("segmentation_challenge_data/train-volume.tif")
    training_label_volume = Image.open("segmentation_challenge_data/train-labels.tif")

    images = []
    labels = []
    for image in ImageSequence.Iterator(training_image_volume):
        images.append(image.copy())

    for label in ImageSequence.Iterator(training_label_volume):
        labels.append(label.copy())

    augment_dataset(factor, images, labels, train_output_path, label_output_path)


def augment_phc_dataset(factor):
    train_output_path = f'cell_tracking_challenge_data/PhC-C2DH-U373_train/augmented/train_{factor}x'
    label_output_path = f'cell_tracking_challenge_data/PhC-C2DH-U373_train/augmented/labels_{factor}x'
    makedir(train_output_path)
    makedir(label_output_path)

    images = list(map(Image.open, glob.glob('cell_tracking_challenge_data/PhC-C2DH-U373_train/01/*.tif')))
    images.extend(list(map(Image.open, glob.glob('cell_tracking_challenge_data/PhC-C2DH-U373_train/02/*.tif'))))
    labels = list(map(Image.open, glob.glob('cell_tracking_challenge_data/PhC-C2DH-U373_train/01_GT/*.tif')))
    labels.extend(list(map(Image.open, glob.glob('cell_tracking_challenge_data/PhC-C2DH-U373_train/02_GT/*.tif'))))

    augment_dataset(factor, images, labels, train_output_path, label_output_path)


def augment_dic_dataset(factor):
    train_output_path = f'cell_tracking_challenge_data/DIC-C2DH-HeLa_train/augmented/train_{factor}x'
    label_output_path = f'cell_tracking_challenge_data/DIC-C2DH-HeLa_train/augmented/labels_{factor}x'
    makedir(train_output_path)
    makedir(label_output_path)

    images = list(map(Image.open, glob.glob('cell_tracking_challenge_data/DIC-C2DH-HeLa_train/01/*.tif')))
    images.extend(list(map(Image.open, glob.glob('cell_tracking_challenge_data/DIC-C2DH-HeLa_train/02/*.tif'))))
    labels = list(map(Image.open, glob.glob('cell_tracking_challenge_data/DIC-C2DH-HeLa_train/01_GT/*.tif')))
    labels.extend(list(map(Image.open, glob.glob('cell_tracking_challenge_data/DIC-C2DH-HeLa_train/02_GT/*.tif'))))

    augment_dataset(factor, images, labels, train_output_path, label_output_path)


def augment_dataset(factor, images, labels, train_output_path, label_output_path):
    for i, (image, label) in enumerate(zip(images, labels)):
        for j in range(factor):
            augmented_image, augmented_label = augment_image(image, label)
            Image.fromarray(augmented_image).save(f"{train_output_path}/train_{j}_{i}.tif", "tiff")
            Image.fromarray(augmented_label).save(f"{label_output_path}/label_{j}_{i}.tif", "tiff")


def main():
    augment_segmentation_dataset(500)
    # augment_phc_dataset(1)
    # augment_dic_dataset(1)


if __name__ == "__main__":
    # start_time = time.time()
    main()
    # print("--- %s seconds ---" % (time.time() - start_time))



