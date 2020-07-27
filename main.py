import math
import time
from io import BytesIO

import numpy as np
from PIL import Image

import data


def identification(image_bytes):
    # 灰度处理并创建二维矩阵
    img_matrix = np.array(Image.open(BytesIO(image_bytes)).convert("L"))
    # 获取矩阵（图像）的长宽
    rows, cols = img_matrix.shape
    for i in range(rows):
        for j in range(cols):
            # 与阈值比较
            if img_matrix[i, j] <= 128:
                # 设为灰度最小值
                img_matrix[i, j] = 0
            else:
                # 设为灰度最大值
                img_matrix[i, j] = 1
    # 每行最小值
    row_min = np.min(img_matrix, axis=1)
    # 找到第一个有图像的行
    row_start = np.argmin(row_min)
    # 找到最后一个有图像的行
    row_end = np.argmin(np.flip(row_min))
    # 只取有图像的行
    img_matrix = img_matrix[row_start:-row_end, :]
    codes = [0] * 4
    for i in range(4):
        # 切片
        imag_matrix_spited = img_matrix[:, 19 * i:19 * (i + 1)]
        col_min = np.min(imag_matrix_spited, axis=0)
        col_start = np.argmin(col_min)
        col_end = np.argmin(np.flip(col_min))
        # 图像宽度
        width = col_min.shape[0] - (col_start + col_end)
        # 宽度扩宽到 9 像素
        width_rest = 9 - width
        # 左边界
        col_start -= int(math.ceil(width_rest / 2.0))
        # 右边界
        col_end -= int(math.floor(width_rest / 2.0))
        # 裁剪为 9 像素宽的图像
        imag_matrix_spited = imag_matrix_spited[:, col_start:-col_end]
        res = [0] * 10
        # 展开成一维
        x = imag_matrix_spited.flatten()
        for j in range(10):
            # 一次取字库中标准数据
            y = data.array_map[j]
            # 通过异或计算不同元素的数量
            res[j] = np.sum(x ^ y)
            # 取差异最小的下标
        codes[i] = str(np.argmin(res))
    return ''.join(codes)


if __name__ == "__main__":
    start_time = time.time()
    for i in range(100):
        with open("vcode/" + str(i) + ".png", "rb") as f:
            image = f.read()
        res = identification(image)
        print(res)
    end_time = time.time()
    interval = (end_time - start_time)
    print(interval)
