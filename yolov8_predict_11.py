from ultralytics import YOLO
import cv2
import numpy as np
import os
from tqdm import tqdm

# ===================== 1. 核心配置（所有可调整参数都在这里！） =====================
# ---------------------- 模型/文件路径 ----------------------
MODEL_PATH = "./best.pt"  # 模型绝对路径
LARGE_IMG_PATH = "../01_recombined.bmp"  # 超大图路径
OUTPUT_DIR = "./jiance_output"  # 结果保存路径

# ---------------------- 分割参数 ----------------------
CROP_SIZE = 512  # 小图尺寸（宽/高）
OVERLAP = 256    # 小图重叠像素（左右+上下都生效）
SKIP_TOP_ROWS = 700  # 跳过图片顶部的行数（不需要处理的像素行，设0则不跳过）
CONF_THRESHOLD = 0.2  # 检测置信度阈值
IOU_THRESHOLD = 0.9   # NMS去重的IoU阈值

# ===================== 2. 初始化模型和文件夹 =====================
# 加载模型
model = YOLO(MODEL_PATH)
# 创建结果文件夹
os.makedirs(OUTPUT_DIR, exist_ok=True)
# 读取超大图
img = cv2.imread(LARGE_IMG_PATH)
if img is None:
    raise ValueError(f"无法读取图片：{LARGE_IMG_PATH}")
img_h, img_w = img.shape[:2]
print(f"超大图原始尺寸：{img_w}×{img_h}")

# 跳过顶部指定行数，生成处理后的图片
if SKIP_TOP_ROWS > 0 and SKIP_TOP_ROWS < img_h:
    img = img[SKIP_TOP_ROWS:, :]  # 裁剪掉顶部SKIP_TOP_ROWS行
    img_h = img.shape[0]  # 更新图片高度
    print(f"跳过顶部{SKIP_TOP_ROWS}行后，处理尺寸：{img_w}×{img_h}")
elif SKIP_TOP_ROWS >= img_h:
    raise ValueError(f"跳过的行数({SKIP_TOP_ROWS})超过图片总高度({img_h})！")

# ===================== 3. 超大图分割（修复上下重叠+仅处理指定区域） =====================
def split_large_image(img, crop_size, overlap):
    """
    分割超大图为小图（左右+上下都重叠）
    :param img: 跳过顶部行后的图片
    :param crop_size: 小图尺寸
    :param overlap: 重叠像素（左右/上下统一）
    :return: 小图列表 [(crop_img, x1, y1), ...] （y1是跳过顶部后的相对坐标）
    """
    img_h, img_w = img.shape[:2]
    crops = []
    step = crop_size - overlap  # 实际步长（非重叠部分）

    # 遍历y轴（高度方向：上下重叠生效）
    y = 0
    while y < img_h:
        # 遍历x轴（宽度方向：左右重叠生效）
        x = 0
        while x < img_w:
            # 计算裁剪区域（避免越界）
            x1 = x
            y1 = y
            x2 = min(x1 + crop_size, img_w)
            y2 = min(y1 + crop_size, img_h)

            # 最后一块不足crop_size时，偏移补满（保证小图尺寸一致）
            if x2 - x1 < crop_size:
                x1 = max(0, img_w - crop_size)
                x2 = img_w
            if y2 - y1 < crop_size:
                y1 = max(0, img_h - crop_size)
                y2 = img_h

            # 裁剪小图
            crop = img[y1:y2, x1:x2]
            # 记录小图坐标（y1需要加上跳过的行数，映射回原图）
            original_y1 = y1 + SKIP_TOP_ROWS
            crops.append((crop, x1, original_y1))

            # 移动到下一列（左右步长）
            x += step

        # 移动到下一行（上下步长，重叠生效）
        y += step

    return crops

# 执行分割
crops = split_large_image(img, CROP_SIZE, OVERLAP)
print(f"分割完成，共生成 {len(crops)} 张小图（已跳过顶部{SKIP_TOP_ROWS}行）")

# ===================== 4. 小图批量检测（坐标映射修正） =====================
def detect_crops(model, crops, conf_threshold):
    """检测所有小图，返回检测框（映射回原图坐标）"""
    all_boxes = []  # 存储(类别, 置信度, x1, y1, x2, y2) （原图绝对坐标）
    for i, (crop, x_offset, y_offset) in enumerate(tqdm(crops, desc="检测小图中")):
        # 检测小图
        results = model.predict(
            source=crop,
            conf=conf_threshold,
            imgsz=512,
            save=False,
            show=False,
            verbose=False
        )
        # 处理检测结果
        result = results[0]
        if len(result.boxes) == 0:
            continue
        # 遍历检测框，映射回原图坐标
        for box in result.boxes:
            cls_name = result.names[int(box.cls)]
            conf = box.conf.item()
            # 小图内的坐标
            x1_crop, y1_crop, x2_crop, y2_crop = box.xyxy[0].tolist()
            # 映射到原图坐标（x偏移+左右，y偏移+上下+跳过的行数）
            x1 = x_offset + x1_crop
            y1 = y_offset + y1_crop  # y_offset已包含SKIP_TOP_ROWS
            x2 = x_offset + x2_crop
            y2 = y_offset + y2_crop
            all_boxes.append((cls_name, conf, x1, y1, x2, y2))
    return all_boxes

# 执行检测
all_boxes = detect_crops(model, crops, CONF_THRESHOLD)
print(f"检测完成，共识别到 {len(all_boxes)} 个目标")

# ===================== 5. 非极大值抑制（NMS）去重 =====================
def non_max_suppression(boxes, iou_threshold):
    """对检测框去重，避免重叠区域重复标注"""
    if len(boxes) == 0:
        return []
    
    # 转换为numpy数组
    cls_names = [b[0] for b in boxes]
    confs = np.array([b[1] for b in boxes])
    xyxy = np.array([b[2:] for b in boxes])
    
    # 按置信度排序
    indices = np.argsort(-confs)
    keep = []
    
    while len(indices) > 0:
        # 保留置信度最高的框
        current = indices[0]
        keep.append(current)
        
        # 计算当前框与其他框的IoU
        if len(indices) == 1:
            break
        iou = cv2.dnn.NMSBoxes(
            xyxy[indices[1:]].tolist(),
            confs[indices[1:]].tolist(),
            score_threshold=0.0,
            nms_threshold=iou_threshold
        )
        
        # 更新索引
        if isinstance(iou, (list, np.ndarray)) and len(iou) > 0:
            indices = indices[1:][iou]
        else:
            break
    
    # 返回去重后的框
    return [boxes[i] for i in keep]

# 执行NMS去重
filtered_boxes = non_max_suppression(all_boxes, iou_threshold=IOU_THRESHOLD)
print(f"NMS去重后，剩余 {len(filtered_boxes)} 个目标")

# ===================== 6. 绘制检测框并保存大图（还原原图尺寸） =====================
# 重新读取原图（用于绘制完整标注）
original_img = cv2.imread(LARGE_IMG_PATH)
# 类别颜色映射
color_map = {
    "Foreign_Object": (0, 0, 255),    # 红色：外来物
    "Structual_Defect": (0, 255, 0)   # 绿色：结构缺陷
}

# 绘制检测框（仅在非跳过区域绘制）
for cls_name, conf, x1, y1, x2, y2 in filtered_boxes:
    # 转换为整数坐标
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    # 只绘制在非跳过区域（避免标注到顶部跳过的行）
    if y1 > SKIP_TOP_ROWS:
        # 获取颜色
        color = color_map.get(cls_name, (255, 0, 0))  # 蓝色：未知类别
        # 绘制矩形框
        cv2.rectangle(original_img, (x1, y1), (x2, y2), color, 2)
        # 绘制类别+置信度标签
        label = f"{cls_name} {conf:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_y = y1 - 10 if y1 - 10 > 10 else y1 + label_size[1] + 10
        cv2.rectangle(original_img, (x1, label_y - label_size[1] - 5), 
                      (x1 + label_size[0] + 5, label_y + 5), color, -1)
        cv2.putText(original_img, label, (x1 + 2, label_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# 保存结果大图
save_path = os.path.join(OUTPUT_DIR, "large_image_detected.jpg")
cv2.imwrite(save_path, original_img)
print(f"带标注的超大图已保存：{save_path}")

# ===================== 7. 生成检测报告 =====================
report = f"""
===================== 超大图检测报告 =====================
1. 原图信息：
   - 路径：{LARGE_IMG_PATH}
   - 原始尺寸：{original_img.shape[1]}×{original_img.shape[0]}
   - 跳过顶部行数：{SKIP_TOP_ROWS}
   - 实际处理尺寸：{img_w}×{img_h}
   - 分割小图尺寸：{CROP_SIZE}×{CROP_SIZE}（重叠{OVERLAP}像素，左右+上下都生效）
   - 分割小图数量：{len(crops)} 张

2. 检测结果：
   - 原始检测目标数：{len(all_boxes)} 个
   - NMS去重后目标数：{len(filtered_boxes)} 个
   - 类别统计：
     - Foreign_Object（外来物）：{len([b for b in filtered_boxes if b[0] == "Foreign_Object"])} 个
     - Structual_Defect（结构缺陷）：{len([b for b in filtered_boxes if b[0] == "Structual_Defect"])} 个

3. 结果保存：
   - 带标注的超大图：{save_path}
==========================================================
"""

# 保存报告
with open(os.path.join(OUTPUT_DIR, "detection_report.txt"), "w", encoding="utf-8") as f:
    f.write(report)
print(report)