import cv2
import torch
import numpy as np
from numpy import random

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized
import settings


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def detect_simple_image(path_image,
                        path_save,
                        weights,
                        img_size: int = 640,
                        conf_thres: int = 0.25,
                        iou_thres: int = 0.45,
                        device: str = 'cpu',
                        ):
    classes_to_filter = [
        settings.class_labels]

    opt = {
        "weights": weights,
        # Path to weights file default weights are for nano model
        "img-size": img_size,  # default image size
        "conf-thres": conf_thres,  # confidence threshold for inference.
        "iou-thres": iou_thres,  # NMS IoU threshold for inference.
        "device": device,  # device to run our model i.e. 0 or 0,1,2,3 or cpu
        "classes": classes_to_filter  # list of classes to filter or None
    }

    with torch.no_grad():
        weights, imgsz = opt['weights'], opt['img-size']
        set_logging()
        device = select_device(opt['device'])
        half = device.type != 'cpu'
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if half:
            model.half()

        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

        img0 = cv2.imread(path_image)
        img = letterbox(img0, imgsz, stride=stride)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=False)[0]

        # Apply NMS
        classes = None
        if opt['classes']:
            classes = []
            for class_name in opt['classes'][0]:
                classes.append(names.index(class_name))

        if classes:
            classes = [i for i in range(len(names)) if i not in classes]

        pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes=[0, 1, 2], agnostic=False)
        t2 = time_synchronized()
        for i, det in enumerate(pred):
            s = ''
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()

                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, img0, label=label, color=colors[int(cls)], line_thickness=1)
    print(f"Save image : {path_save}")
    cv2.imwrite(path_save, img0)


def run():
    path_image = 'PATH_TO_YOUR_IMAGE'
    path_save = 'PATH_TO_SAVE_IMAGE'
    weights = 'PATH_TO_WEIGHTS'
    detect_simple_image(path_image, path_save, weights)


if __name__ == '__main__':
    run()
