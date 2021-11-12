import argparse
import os
import shutil
import sys
print(sys.path.append("/content/Segm_LungNodule_CNN_kmeans_knn"))

from config.paths import base_dataset_dir

path_to_original_images_ = "original_training"
path_to_original_labels_ = "original_training_mask"

path_to_validation_images_ = "original_validation"
path_to_validation_labels_ = "original_validation_mask"


def run(path_to_original_images, path_to_original_labels,
        path_to_validation_images, path_to_validation_labels):
    path_to_original_images = os.path.join(base_dataset_dir, path_to_original_images)
    path_to_original_labels = os.path.join(base_dataset_dir, path_to_original_labels)
    path_to_validation_images = os.path.join(base_dataset_dir, path_to_validation_images)
    path_to_validation_labels = os.path.join(base_dataset_dir, path_to_validation_labels)

    original_length = len(os.listdir(path_to_original_images))
    original_length_l = len(os.listdir(path_to_original_labels))

    os.makedirs(path_to_validation_images, exist_ok=True)
    os.makedirs(path_to_validation_labels, exist_ok=True)

    validation_labels = ['sub-verse041_dir-ax_seg-vert_msk.nii.gz', 'sub-verse042_dir-ax_seg-vert_msk.nii.gz', 'sub-verse043_dir-iso_seg-vert_msk.nii.gz', 'sub-verse044_dir-ax_seg-vert_msk.nii.gz', 'sub-verse045_dir-ax_seg-vert_msk.nii.gz', 'sub-verse046_dir-ax_seg-vert_msk.nii.gz']
    validation_images = [vl.replace('_seg-vert_msk', '_ct') for vl in validation_labels]

    assert len(validation_images) == len(validation_labels), "Mismatch in sizes between validation images and labels"

    trainval_images = os.listdir(path_to_original_images)
    trainval_labels = os.listdir(path_to_original_labels)

    trainval_images.sort()
    trainval_labels.sort()

    cnt_val = 0

    for trainval_image, trainval_label in zip(trainval_images, trainval_labels):
        is_val_i = trainval_image in validation_images
        is_val_l = trainval_label in validation_labels
        print("{} {} {}{}".format(trainval_image,is_val_i, trainval_label, is_val_l))
        assert is_val_i == is_val_l, "Mismatch between validation image and label! {} and {}".format(trainval_image, trainval_label)

        if is_val_i:
            cnt_val += 1
            source_image = os.path.join(path_to_original_images, trainval_image)
            destination_image = os.path.join(path_to_validation_images, trainval_image)
            shutil.move(source_image, destination_image)
            print("moving {}".format(destination_image))

            source_label = os.path.join(path_to_original_labels, trainval_label)
            destination_label = os.path.join(path_to_validation_labels, trainval_label)
            shutil.move(source_label, destination_label)

    print("Counter Val  = ", cnt_val)
    print("Validation I = ", len(validation_images))
    print("Validation L = ", len(validation_labels))
    assert cnt_val == len(validation_images) == len(validation_labels), "Mismatch in size between cnt_val and validation set"

    train_len_i = len(os.listdir(path_to_original_images))
    val_len_i = len(os.listdir(path_to_validation_images))

    assert train_len_i + val_len_i == original_length, "Mismatch after train-val split"

    train_len_l = len(os.listdir(path_to_original_labels))
    val_len_l = len(os.listdir(path_to_validation_labels))

    assert train_len_l + val_len_l == original_length_l, "Mismatch in labels after train-val split"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for performing trainval/test splitting")
    parser.add_argument(
        "-oti",
        "--original-training-images",
        default=path_to_original_images_, type=str,
        help="Specify the path where there are the original images"
    )
    parser.add_argument(
        "-otl",
        "--original-training-labels",
        default=path_to_original_labels_, type=str,
        help="Specify the path where there are the original labels"
    )
    parser.add_argument(
        "-ovi",
        "--original-validation-images",
        default=path_to_validation_images_, type=str,
        help="Specify the path where to put the original validation images"
    )
    parser.add_argument(
        "-ovl",
        "--original-validation-labels",
        default=path_to_validation_labels_, type=str,
        help="Specify the path where to put the original validation labels"
    )
    args = parser.parse_args()
    run(
        args.original_training_images,
        args.original_training_labels,
        args.original_validation_images,
        args.original_validation_labels
    )
