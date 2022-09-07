import cv2
import torch
from numpy import random

import settings


def plot_one_box(x, img, color=None, label=None, line_thickness=1):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def detect_one_img(path_img: str, path_save: str, model: torch.Model, show):
    """
    Predict the location of objects and visualize them in the input image
    :param path_img: the path to the image to predict
    :param path_save: the path for saving the image with found objects in boxes
    :param model: box prediction model
    :param show: show the result or not
    :return:
    """
    img = cv2.imread(path_img)
    res_pred = model(path_img).pred[0].numpy()
    for obj in res_pred:
        plot_one_box(obj[:-2], img, label=settings.class_labels[int(obj[-1])])
    if show:
        cv2.imshow('pred image', img)
        cv2.waitKey()
        cv2.destroyAllWindows()

    print(f"Save image : {path_save}")
    cv2.imwrite(path_save, img)


def run():
    path_image = 'PATH_TO_YOUR_IMAGE'
    path_save = 'PATH_TO_SAVE_IMAGE'
    weights = 'PATH_TO_WEIGHTS'
    model_yolo = torch.hub.load(r'WongKinYiu/yolov7', 'custom', weights)
    detect_one_img(path_image, path_save, model_yolo, True)


if __name__ == '__main__':
    run()
