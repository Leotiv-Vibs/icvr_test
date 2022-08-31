def get_box_data(path_txt_file: str, w_img: int, h_img: int):
    with open(path_txt_file) as f:
        lines = f.readlines()
    objs_info = []
    for line in lines:
        obj = list(map(float, line.split(" ")))
        objs_info.append([int(obj[0]), int(obj[1] * w_img), int(obj[2] * h_img), int(obj[3] * w_img),
                         int(obj[4] * h_img)])
    return objs_info