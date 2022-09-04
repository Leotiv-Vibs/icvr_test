import re
import os
from typing import List

import albumentations as A
import cv2
from tqdm import tqdm

import tools
import settings


class DataAugmentation:
    """
    Class for data augmentation
    """
    def __init__(self, path_data, path_save_img, path_save_lbl):
        self.path_data = path_data
        self.path_save_img = path_save_img
        self.path_save_lbl = path_save_lbl

        self.name_files = os.listdir(self.path_data)
        self.txt_data = sorted(list(filter(re.compile(".*txt").match, self.name_files)))
        self.img_data = sorted(list(filter(re.compile(".*jpg").match, self.name_files)))

        self.list_transforms = self.get_list_transforms()

    def create_augmentation(self):
        """
        perform augmentation for each image in the dataset and save the new image and new label
        :return:
        """
        for img in tqdm(self.img_data, desc='Augmentation image'):
            image = cv2.imread(f'{self.path_data}\{img}')
            bbox = tools.bbox_yolo_format(f'{self.path_data}\{img[:-4]}.txt')
            for trans in self.list_transforms:
                image = image.copy()
                try:
                    transform = trans(image=image, bboxes=bbox, class_labels=settings.class_labels)
                except:
                    print(f'Bad data: {img}')
                    continue
                image_ = transform['image']
                bboxes = transform['bboxes']
                bboxes = tools.format_train_label(bboxes)
                count_file = len(os.listdir(self.path_save_img))
                cv2.imwrite(f'{self.path_save_img}\\aug_{count_file}.jpg', image_)
                tools.write_train_label(f'{self.path_save_lbl}\\aug_{count_file}.txt', bboxes)

    @staticmethod
    def get_list_transforms() -> List[A.Compose]:
        """
        create a list of objects to augment images
        :return:
        """
        transform_hor_flip = A.Compose([
            A.HorizontalFlip(always_apply=True),
        ], bbox_params=A.BboxParams(format='yolo'))

        transform_random_gamma = A.Compose([
            A.RandomGamma(gamma_limit=(50, 100), p=0.5, always_apply=True),
        ], bbox_params=A.BboxParams(format='yolo'))

        transform_ver_flip = A.Compose([
            A.VerticalFlip(always_apply=True),
        ], bbox_params=A.BboxParams(format='yolo'))

        transform_gaus_blur = A.Compose([
            A.GaussianBlur(blur_limit=(3, 7), always_apply=True, p=0.5),
        ], bbox_params=A.BboxParams(format='yolo'))

        transform_gaus_noise = A.Compose([
            A.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=True, p=0.5),
        ], bbox_params=A.BboxParams(format='yolo'))

        transform_median_blur = A.Compose([
            A.MedianBlur(blur_limit=7, always_apply=True, p=0.5),
        ], bbox_params=A.BboxParams(format='yolo'))

        transform_random_contr = A.Compose([
            A.RandomContrast(limit=0.5, always_apply=True, p=0.5),
        ], bbox_params=A.BboxParams(format='yolo'))

        transform_hue_sat_val = A.Compose([
            A.HueSaturationValue(always_apply=True),
        ], bbox_params=A.BboxParams(format='yolo'))

        transform_rot_45 = A.Compose([
            A.Rotate(limit=45, always_apply=True),
        ], bbox_params=A.BboxParams(format='yolo'))

        transform_rot_90 = A.Compose([

            A.Rotate(limit=90, always_apply=True),
        ], bbox_params=A.BboxParams(format='yolo'))

        list_transforms = [transform_hor_flip,
                           transform_random_gamma,
                           transform_ver_flip,
                           # transform_gaus_blur,
                           # transform_gaus_noise,
                           # transform_median_blur,
                           transform_random_contr,
                           transform_hue_sat_val,
                           # transform_rot_45,
                           # transform_rot_90
                           ]

        return list_transforms


def run():
    path_data = r'C:\Users\79614\Desktop\train'
    path_save_img = r'C:\Users\79614\PycharmProjects\icvr_test\aug_data_img'
    path_save_lbl = r'C:\Users\79614\PycharmProjects\icvr_test\aug_data_lbl'
    dataAugment = DataAugmentation(path_data, path_save_img, path_save_lbl)
    dataAugment.create_augmentation()


if __name__ == '__main__':
    run()
