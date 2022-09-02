import os
import re
import shutil

from tqdm import tqdm

import settings


class PrepareDirectoryTrain:
    def __init__(self,
                 path_data_img,
                 path_data_lbl,
                 path_create,
                 number_class,
                 labels_class,
                 train_size: int = 0.7,
                 valid_size: int = 0.2,
                 test_size: int = 0.1,
                 ):
        self.path_data_img = path_data_img
        self.path_data_lbl = path_data_lbl
        self.path_create = path_create
        self.number_class = number_class
        self.labels_class = labels_class

        self.path_train = f'{self.path_create}/train'
        self.path_valid = f'{self.path_create}/valid'
        self.path_test = f'{self.path_create}/test'

        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size

        self.file_extension = 'jpg'

    def __call__(self, *args, **kwargs):
        train_img, valid_img, test_img = self.train_valid_test_split(self.path_data_img, self.train_size,
                                                                     self.valid_size, self.test_size)
        self.create_yolo_dir(self.path_create)
        self.prepare_dir_train(self.path_train, self.path_valid, self.path_test, train_img, valid_img, test_img)
        self.create_yaml_file(self.path_train, self.path_valid, self.number_class, self.labels_class)

    def train_valid_test_split(self, path_data, train_size, valid_size, test_size):
        """

        :param path_data:
        :param train_size:
        :param valid_size:
        :param test_size:
        :return:
        """
        name_files = os.listdir(path_data)
        img_data = sorted(list(filter(re.compile(f".*{self.file_extension}").match, name_files)))
        size_data = len(img_data)
        train_img = img_data[:int(size_data * train_size)]
        valid_img = img_data[int(size_data * train_size):int(size_data * train_size) + int(valid_size * size_data)]
        test_img = img_data[int(size_data * train_size) + int(valid_size * size_data):]

        return train_img, valid_img, test_img

    def prepare_dir_train(self, path_train, path_valid, path_test, train_img, valid_img, test_img):
        """

        :param path_train:
        :param path_valid:
        :param path_test:
        :param train_img:
        :param valid_img:
        :param test_img:
        :return:
        """
        for tr in tqdm(train_img):
            shutil.copy2(f'{self.path_data_img}/{tr}', f'{path_train}/images/{tr}')
            shutil.copy2(f'{self.path_data_lbl}/{tr[:-4]}.txt', f'{path_train}/labels/{tr[:-4]}.txt')

        for vl in tqdm(valid_img):
            shutil.copy2(f'{self.path_data_img}/{vl}', f'{path_valid}/images/{vl}')
            shutil.copy2(f'{self.path_data_lbl}/{vl[:-4]}.txt', f'{path_valid}/labels/{vl[:-4]}.txt')

        for ts in tqdm(test_img):
            shutil.copy2(f'{self.path_data_img}/{ts}', f'{path_test}/images/{ts}')
            shutil.copy2(f'{self.path_data_lbl}/{ts[:-4]}.txt', f'{path_test}/labels/{ts[:-4]}.txt')

    def create_yaml_file(self, path_train, path_valid, number_class, class_labels):
        train_dir = f'train: {path_train}/images'
        valid_dir = f'valid: {path_valid}/images'

        nc = f'nc: {number_class}'
        labels = f'names: {class_labels}'

        txt_file = '\n'.join([train_dir, valid_dir, nc, labels])
        with open(f'{self.path_create}/data.yaml', 'w') as f:
            f.writelines(txt_file)

    @staticmethod
    def create_yolo_dir(path_create):
        """
        Creating directories for training
        :param path_create:
        :return:
        """
        os.makedirs(f'{path_create}/train/images', exist_ok=True)
        os.makedirs(f'{path_create}/train/labels', exist_ok=True)

        os.makedirs(f'{path_create}/valid/images', exist_ok=True)
        os.makedirs(f'{path_create}/valid/labels', exist_ok=True)

        os.makedirs(f'{path_create}/test/images', exist_ok=True)
        os.makedirs(f'{path_create}/test/labels', exist_ok=True)


def run():
    path_data_img = r'C:\Users\79614\PycharmProjects\icvr_test\aug_data_img'
    path_data_lbl = r'C:\Users\79614\PycharmProjects\icvr_test\aug_data_lbl'

    path_create = r'C:\Users\79614\PycharmProjects\icvr_test\data'
    number_class = settings.number_class
    labels_class = settings.class_labels

    pre_dir_train = PrepareDirectoryTrain(path_data_img, path_data_lbl, path_create, number_class, labels_class)
    pre_dir_train()


if __name__ == '__main__':
    run()
