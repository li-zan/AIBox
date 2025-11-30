import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_chinese_label(frame, label, x1, y1, scale=0.7, thickness=2, color=(0, 255, 0)):
    """
    在OpenCV图像上绘制中文标签，并添加底色增强可读性
    该函数返回一个新的图像，不修改原图像。
    Args:
        frame (ndarray): OpenCV图像帧
        label (str): 要绘制的文字
        x1 (int): 左上角x坐标
        y1 (int): 左上角y坐标
        scale (float): 字体缩放比例
        thickness (int): 字体粗细（近似控制）
        color (tuple): 颜色，BGR格式
    Returns:
        ndarray: 绘制好标签的图像
    """
    # 转为PIL图像
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 字体
    font_size = int(25 * scale + thickness * 2)
    font = ImageFont.truetype("./font/msyh.ttf", font_size)

    # BGR → RGB
    color_rgb = (color[2], color[1], color[0])

    # 文字尺寸
    bbox = font.getbbox(label)  # (left, top, right, bottom)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # 标签底色
    bg_color = (0, 0, 0, 160)  # 半透明黑色

    pad = 4
    rect_x1 = x1
    rect_y1 = max(0, y1 - text_h - 10)
    rect_x2 = x1 + text_w + pad * 2
    rect_y2 = rect_y1 + text_h + pad * 4

    # 绘制底色矩形
    draw.rectangle([rect_x1, rect_y1, rect_x2, rect_y2], fill=bg_color)

    # 绘制文字
    draw.text((rect_x1 + pad, rect_y1 + pad), label, font=font, fill=color_rgb)

    # 转回OpenCV格式
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_chinese_label_inplace(frame, label, x1, y1, scale=0.7, thickness=2, color=(0, 255, 0)):
    """
    在OpenCV图像上原地绘制中文标签，并添加底色增强可读性
    该函数直接修改传入的图像帧。
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

    # 字体
    font_size = int(25 * scale + thickness * 2)
    font = ImageFont.truetype("./font/msyh.ttf", font_size)

    # BGR → RGB
    color_rgb = (color[2], color[1], color[0])

    # 文字尺寸
    bbox = font.getbbox(label)  # (left, top, right, bottom)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # 标签底色
    bg_color = (0, 0, 0, 160)  # 半透明黑色

    pad = 4
    rect_x1 = x1
    rect_y1 = max(0, y1 - text_h - 10)
    rect_x2 = x1 + text_w + pad * 2
    rect_y2 = rect_y1 + text_h + pad * 4

    # 绘制底色矩形
    draw.rectangle([rect_x1, rect_y1, rect_x2, rect_y2], fill=bg_color)

    # 绘制文字
    draw.text((rect_x1 + pad, rect_y1 + pad), label, font=font, fill=color_rgb)

    # 将PIL绘制结果覆盖回frame（原地写回）
    frame[:, :, :] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
