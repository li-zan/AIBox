import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_chinese_label(frame, label, x1, y1, scale=0.7, thickness=2, color=(0, 255, 0)):
    """在OpenCV图像上绘制中文标签

    Args:
        frame (ndarray): OpenCV图像帧
        label (str): 要绘制的文字
        x1 (int): 左上角x坐标
        y1 (int): 左上角y坐标
        scale (float): 字体缩放比例
        thickness (int): 字体粗细（近似控制）
        color (tuple): 颜色，BGR格式
    """
    # 转为PIL图像
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 字体大小根据scale和thickness近似估计
    font_size = int(25 * scale + thickness * 2)
    font = ImageFont.truetype("font/msyh.ttf", font_size)

    # 将BGR转为RGB
    color_rgb = (color[2], color[1], color[0])

    # 绘制文字
    draw.text((x1, y1 - 10), label, font=font, fill=color_rgb)

    # 转回OpenCV格式
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_chinese_label_inplace(frame, label, x1, y1, scale=0.7, thickness=2, color=(0, 255, 0)):
    """在OpenCV图像上直接绘制中文标签（原地修改）

    Args:
        frame (ndarray): OpenCV图像帧
        label (str): 要绘制的文字
        x1 (int): 左上角x坐标
        y1 (int): 左上角y坐标
        scale (float): 字体缩放比例
        thickness (int): 字体粗细（近似控制）
        color (tuple): 颜色，BGR格式
    """
    # 计算字体大小
    font_size = int(25 * scale + thickness * 2)
    font = ImageFont.truetype("font/msyh.ttf", font_size)

    # 将frame转为PIL图像（注意这里要copy，否则会影响frame的数据布局）
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 转为RGB颜色
    color_rgb = (color[2], color[1], color[0])

    # 绘制文字
    draw.text((x1, y1 - 10), label, font=font, fill=color_rgb)

    # 将PIL绘制结果覆盖回frame（原地写回）
    frame[:, :, :] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
