from typing import List

import settings

def get_box_data(path_txt_file: str, w_img: int, h_img: int) -> List[List]:
    """
    from a text file to a list of lists px coordinates
    :param path_txt_file: the path to the text data with information about the object
    :param w_img: image width
    :param h_img: image height
    :return: List of lists with object coordinates in pixels
    """
    with open(path_txt_file) as f:
        lines = f.readlines()
    objs_info = []
    for line in lines:
        obj = list(map(float, line.split(" ")))
        objs_info.append([int(obj[0]), int(obj[1] * w_img), int(obj[2] * h_img), int(obj[3] * w_img),
                          int(obj[4] * h_img)])
    return objs_info


def bbox_yolo_format(path_txt_file: str) -> List[List]:
    """
    from yolo format to albumentations format
    :param path_txt_file: the path to the text data with information about the object
    :return: List of lists for augmentations with albumentations
    """
    with open(path_txt_file) as f:
        lines = f.readlines()
    objs_info = []
    for line in lines:
        obj = list(map(float, line.split(" ")))
        if int(obj[0]) == 0:
            obj.append('uniform')
        if int(obj[0]) == 1:
            obj.append('helmet')
        if int(obj[0]) == 2:
            obj.append('human')
        objs_info.append(obj[1:])
    return objs_info


def format_train_label(bboxes: List[List]):
    """
    Function for processing output albumentations into yolo format
    :param bboxes: list of boxes with classes and coordinates
    :return: List of lists in yolo format
    """
    for id in range(len(bboxes)):
        if bboxes[id][-1] == settings.class_labels[0]:
            bboxes[id] = list(bboxes[id][:-1])
            bboxes[id].insert(0, 0)
        elif bboxes[id][-1] == settings.class_labels[1]:
            bboxes[id] = list(bboxes[id][:-1])
            bboxes[id].insert(0, 1)
        elif bboxes[id][-1] == settings.class_labels[2]:
            bboxes[id] = list(bboxes[id][:-1])
            bboxes[id].insert(0, 2)
    return bboxes


def write_train_label(path_save: str, bboxes: List[List]):
    """
    write to the label file in yolo format
    :param path_save: path to save labels
    :param bboxes: list of object lists
    :return:
    """
    with open(path_save, 'w') as file:
        for bbox in bboxes:
            file.writelines(" ".join(map(str, bbox)) + '\n')


def bbox_px_format_list(lines: str, w_img: int, h_img: int) -> List[List]:
    """

    :param lines:
    :param w_img:
    :param h_img:
    :return:
    """
    objs_info = []
    for line in lines:
        objs_info.append([int(line[0]), int(line[1] * w_img), int(line[2] * h_img), int(line[3] * w_img),
                          int(line[4] * h_img)])
    return objs_info
