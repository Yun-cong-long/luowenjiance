"""
图像处理工具包
包含三个核心功能：
1. 按比例压缩图像
2. 模板匹配并计算比例
3. 图像分割重组
"""

import cv2
import numpy as np
import os
from pathlib import Path

# ==================== 图像压缩函数 ====================
def compress_image(input_path, scale_percent, output=False, output_suffix="_compressed"):
    """
    按照输入比例压缩图片
    
    参数:
        input_path: 输入图片路径（字符串或Path对象）
        scale_percent: 压缩比例（百分比，如50表示压缩到50%）
        output: 是否输出压缩后图像（默认False）
        output_suffix: 输出文件名的后缀（默认"_compressed"）
    
    返回:
        压缩后的图像数组（NumPy数组）
    
    示例:
        img = compress_image("photo.jpg", 50, output=True)
    """
    # 读取图像
    img = cv2.imread(str(input_path))
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {input_path}")
    
    # 计算新的尺寸
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    
    # 缩放图像
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    
    # 如果要求输出，保存到原目录
    if output:
        path_obj = Path(input_path)
        new_filename = f"{path_obj.stem}{output_suffix}{path_obj.suffix}"
        output_path = path_obj.parent / new_filename
        cv2.imwrite(str(output_path), resized)
    
    return resized

# ==================== 模板匹配函数 ====================
def template_matching(large_image, template_path, threshold=0.8, draw_match=False, save_path=None):
    """
    模板匹配函数：在小图片中匹配大图片中的图像
    
    参数:
        large_image: 大图片路径(**12.31修改：大图片文件，而非地址**)
        template_path: 模板小图片路径
        threshold: 匹配阈值（0-1之间，默认0.8）
        draw_match: 是否绘制匹配框并保存（默认False）
        save_path: 保存匹配结果的路径（当draw_match=True时必须提供）
    
    返回:
        tuple: (left_x, left_ratio, right_ratio)
        - left_x: 匹配框左侧横坐标
        - left_ratio: 左侧区域占总宽度的比例
        - right_ratio: 右侧区域占总宽度的比例
    
    示例:
        x, left_ratio, right_ratio = template_matching("big.jpg", "small.jpg")
    """
    # 读取图像
    img = large_image
    template = cv2.imread(str(template_path))
    
    if img is None or template is None:
        raise ValueError("无法读取图像文件")
    
    # 获取模板尺寸
    h, w = template.shape[:2]
    
    # 使用模板匹配
    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    
    # 寻找最佳匹配位置
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # 如果匹配度低于阈值，返回None
    if max_val < threshold:
        return None, None, None
    
    # 计算匹配框的左侧横坐标
    left_x = max_loc[0]
    
    # 计算整个图像的宽度
    total_width = img.shape[1]
    
    # 计算左右图像横向所占比例
    left_ratio = left_x / total_width
    right_ratio = (total_width - left_x) / total_width
    
    # 如果需要绘制匹配框
    if draw_match:
        if save_path is None:
            raise ValueError("当draw_match=True时，必须提供save_path参数")
        
        img_copy = large_image.copy()
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(img_copy, top_left, bottom_right, (0, 255, 0), 2)
        
        # 保存标记后的图像
        cv2.imwrite(str(save_path), img_copy)
    
    return left_x, left_ratio, right_ratio

# ==================== 图像分割重组函数 ====================
def image_split_recombine(input_path, split_ratio=0.5, save_output=False, output_path=None, output_suffix="_recombined"):
    """
    图像分割重组函数：按照比例纵向分割图像，交换位置后重新拼接
    
    参数:
        input_path: 输入图片路径
        split_ratio: 分割比例（0-1之间），表示左侧部分的比例
        save_output: 是否保存输出图像（默认False）
        output_path: 输出图像路径（如果为None则自动生成）
        output_suffix: 输出文件名的后缀（默认"_recombined"）
    
    返回:
        tuple: (recombined_image, output_path)
        - recombined_image: 重组后的图像数组
        - output_path: 输出文件的路径（如果未保存则为None）
    
    示例:
        img, path = image_split_recombine("photo.jpg", 0.3, save_output=True)
    """
    # 读取图像
    img = cv2.imread(str(input_path))
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {input_path}")
    
    # 获取图像尺寸
    height, width = img.shape[:2]
    
    # 计算分割点
    split_point = int(width * split_ratio)
    
    # 分割图像
    left_part = img[:, :split_point]
    right_part = img[:, split_point:]
    
    # 交换位置并重新拼接
    recombined = np.hstack([right_part, left_part])
    
    # 如果需要保存输出
    final_output_path = None
    if save_output:
        # 如果未指定输出路径，则自动生成
        if output_path is None:
            path_obj = Path(input_path)
            final_output_path = path_obj.parent / f"{path_obj.stem}{output_suffix}{path_obj.suffix}"
        else:
            final_output_path = Path(output_path)
        
        # 确保输出目录存在
        final_output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存图像
        cv2.imwrite(str(final_output_path), recombined)
        print(f"图像已保存至: {final_output_path}")
    
    return recombined, str(final_output_path) if final_output_path else None


# ==================== 辅助函数 ====================
def show_image(image, window_name="Image", wait_time=0):
    """
    显示图像（辅助函数）
    
    参数:
        image: 要显示的图像数组
        window_name: 窗口名称
        wait_time: 等待时间（0表示无限等待）
    """
    cv2.imshow(window_name, image)
    cv2.waitKey(wait_time)
    cv2.destroyAllWindows()