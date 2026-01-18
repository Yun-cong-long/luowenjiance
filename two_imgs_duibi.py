"""
基于YOLOv8的目标检测与对比系统 - 优化版
功能：预处理 → 滑动窗口检测 → 全局去重 → 目标对比
针对异物(Foreign_Object)和缺陷(Structual_Defect)的检测
"""

import argparse
import cv2
import numpy as np
import torch
import os
from pathlib import Path
from ultralytics import YOLO
from image_utils import process_image_pipeline
import yaml
from typing import List, Tuple, Dict, Any, Optional
import time
from dataclasses import dataclass
import json
from collections import defaultdict

@dataclass
class Detection:
    """检测结果数据类"""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str
    source: str = "window"  # window, refined, merged
    
    @property
    def center(self) -> Tuple[float, float]:
        """返回边界框中心点"""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def area(self) -> float:
        """返回边界框面积"""
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    @property
    def width(self) -> float:
        """返回宽度"""
        return self.x2 - self.x1
    
    @property
    def height(self) -> float:
        """返回高度"""
        return self.y2 - self.y1
    
    @property
    def aspect_ratio(self) -> float:
        """返回宽高比"""
        if self.height == 0:
            return float('inf')
        return self.width / self.height
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'bbox': [self.x1, self.y1, self.x2, self.y2],
            'confidence': self.confidence,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'center': self.center,
            'area': self.area,
            'aspect_ratio': self.aspect_ratio,
            'source': self.source
        }

class DetectionPostProcessor:
    """检测后处理器，用于处理重叠框、合并框等"""
    
    @staticmethod
    def merge_overlapping_detections(detections: List[Detection], 
                                   iou_threshold: float = 0.3,
                                   max_aspect_ratio: float = 3.0) -> List[Detection]:
        """
        合并重叠的检测框，优先保留面积大的框
        
        参数:
            detections: 检测结果列表
            iou_threshold: 合并IoU阈值
            max_aspect_ratio: 最大宽高比，超过此值的认为是长条形目标
        
        返回:
            List[Detection]: 合并后的检测结果
        """
        if not detections:
            return []
        
        # 按类别分组处理
        detections_by_class = defaultdict(list)
        for det in detections:
            detections_by_class[det.class_id].append(det)
        
        merged_detections = []
        
        for class_id, class_dets in detections_by_class.items():
            # 按面积降序排序
            class_dets.sort(key=lambda x: x.area, reverse=True)
            
            # 长条形目标单独处理
            normal_dets = [d for d in class_dets if d.aspect_ratio <= max_aspect_ratio]
            long_dets = [d for d in class_dets if d.aspect_ratio > max_aspect_ratio]
            
            # 处理正常比例的目标
            kept_indices = set()
            
            for i in range(len(normal_dets)):
                if i in kept_indices:
                    continue
                
                current_det = normal_dets[i]
                to_merge = []
                
                for j in range(i + 1, len(normal_dets)):
                    if j in kept_indices:
                        continue
                    
                    other_det = normal_dets[j]
                    iou = DetectionPostProcessor.calculate_iou(current_det, other_det)
                    
                    if iou > iou_threshold:
                        to_merge.append(j)
                
                # 如果有重叠的框，选择面积最大的
                if to_merge:
                    # 当前框已经是面积最大的（因为排序过）
                    for idx in to_merge:
                        kept_indices.add(idx)
                
                # 保留当前框
                current_det.source = "merged"
                merged_detections.append(current_det)
                kept_indices.add(i)
            
            # 处理长条形目标（更容易出现多个框框住同一目标）
            for long_det in long_dets:
                # 检查是否已经被正常框覆盖
                is_covered = False
                for kept_det in merged_detections:
                    if kept_det.class_id == class_id:
                        iou = DetectionPostProcessor.calculate_iou(kept_det, long_det)
                        if iou > iou_threshold * 0.7:  # 降低阈值以更好地合并长条形目标
                            is_covered = True
                            break
                
                if not is_covered:
                    long_det.source = "merged_long"
                    merged_detections.append(long_det)
        
        print(f"合并后检测框数量: {len(merged_detections)} (原始: {len(detections)})")
        return merged_detections
    
    @staticmethod
    def calculate_iou(det1: Detection, det2: Detection) -> float:
        """计算两个检测框的IoU"""
        # 计算交集
        inter_x1 = max(det1.x1, det2.x1)
        inter_y1 = max(det1.y1, det2.y1)
        inter_x2 = min(det1.x2, det2.x2)
        inter_y2 = min(det1.y2, det2.y2)
        
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        
        # 计算并集
        area1 = det1.area
        area2 = det2.area
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    @staticmethod
    def calculate_center_distance(det1: Detection, det2: Detection) -> float:
        """计算两个检测框中心点距离"""
        center1 = det1.center
        center2 = det2.center
        return np.sqrt((center2[0] - center1[0])**2 + (center2[1] - center1[1])**2)

class SlidingWindowDetector:
    """滑动窗口检测器，处理大图像"""
    
    def __init__(self, model_path: str, window_size: int = 512, 
                 stride: int = 256, ignore_top: int = 700,
                 conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        初始化检测器
        """
        self.model = YOLO(model_path)
        self.window_size = window_size
        self.stride = stride
        self.ignore_top = ignore_top
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")
        print(f"模型类别: {self.model.names}")
        
    def sliding_window_detection(self, image: np.ndarray) -> List[Detection]:
        """
        滑动窗口检测
        
        返回:
            List[Detection]: 检测结果列表
        """
        height, width = image.shape[:2]
        all_detections = []
        
        # 忽略顶部区域
        start_y = self.ignore_top
        
        print(f"图像尺寸: {width}x{height}, 窗口大小: {self.window_size}, 步长: {self.stride}")
        print(f"忽略顶部 {self.ignore_top} 像素")
        
        # 计算窗口数量
        num_windows_y = (height - start_y - self.window_size) // self.stride + 1
        num_windows_x = (width - self.window_size) // self.stride + 1
        total_windows = num_windows_y * num_windows_x
        
        print(f"总窗口数: {total_windows} ({num_windows_x}x{num_windows_y})")
        
        window_count = 0
        det_count = 0
        
        # 滑动窗口
        for y in range(start_y, height - self.window_size + 1, self.stride):
            for x in range(0, width - self.window_size + 1, self.stride):
                window_count += 1
                if window_count % 100 == 0:
                    print(f"处理窗口中... {window_count}/{total_windows}，已检测到 {det_count} 个目标")
                
                # 提取窗口
                window = image[y:y+self.window_size, x:x+self.window_size]
                
                # YOLOv8检测
                try:
                    results = self.model(window, conf=self.conf_threshold, 
                                       iou=self.iou_threshold, device=self.device, verbose=False)
                    
                    # 处理检测结果
                    for result in results:
                        boxes = result.boxes
                        if boxes is not None and len(boxes) > 0:
                            for box in boxes:
                                # 获取检测信息
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                conf = box.conf[0].cpu().numpy()
                                cls_id = int(box.cls[0].cpu().numpy())
                                
                                # 转换为原图坐标
                                x1_global = x + x1
                                y1_global = y + y1
                                x2_global = x + x2
                                y2_global = y + y2
                                
                                detection = Detection(
                                    x1=float(x1_global),
                                    y1=float(y1_global),
                                    x2=float(x2_global),
                                    y2=float(y2_global),
                                    confidence=float(conf),
                                    class_id=cls_id,
                                    class_name=self.model.names[cls_id],
                                    source="window"
                                )
                                all_detections.append(detection)
                                det_count += 1
                except Exception as e:
                    print(f"窗口({x},{y})检测错误: {e}")
                    continue
        
        print(f"滑动窗口检测完成，初步检测到 {len(all_detections)} 个目标")
        return all_detections

class EnhancedDetectionComparator:
    """增强版检测结果比较器，支持死区范围"""
    
    def __init__(self, class_names: Dict[int, str], 
                 dead_zone: float = 20.0,  # 死区范围，单位像素
                 match_distance: float = 100.0):  # 匹配距离阈值
        self.class_names = class_names
        self.dead_zone = dead_zone
        self.match_distance = match_distance
        
    def compare(self, old_detections: List[Detection], 
                new_detections: List[Detection]) -> Dict[str, Any]:
        """
        比较新旧检测结果
        
        根据需求：
        - 异物(Foreign_Object)会新增和消失
        - 缺陷(Structual_Defect)只会新增
        - 死区内位置波动视为未变化
        - 不统计移动目标
        
        返回:
            Dict: 包含对比结果的字典
        """
        comparison = {
            'summary': {
                'old_total': len(old_detections),
                'new_total': len(new_detections),
                'total_difference': len(new_detections) - len(old_detections)
            },
            'by_class': {},
            'added': [],
            'removed': [],
            'unchanged': [],
            'dead_zone_unchanged': [],  # 死区内视为未变化的目标
            'moved_but_ignored': []  # 移动但被忽略的目标（不统计）
        }
        
        # 按类别统计
        for cls_id, cls_name in self.class_names.items():
            old_cls = [d for d in old_detections if d.class_id == cls_id]
            new_cls = [d for d in new_detections if d.class_id == cls_id]
            
            comparison['by_class'][cls_name] = {
                'old_count': len(old_cls),
                'new_count': len(new_cls),
                'difference': len(new_cls) - len(old_cls),
                'added_count': 0,
                'removed_count': 0
            }
        
        # 匹配检测结果
        matched_old = []
        matched_new = []
        
        # 优先匹配同一类别的检测
        for i, old_det in enumerate(old_detections):
            best_match_idx = -1
            best_match_distance = float('inf')
            
            for j, new_det in enumerate(new_detections):
                if j in matched_new:
                    continue
                
                # 必须是同一类别
                if old_det.class_id != new_det.class_id:
                    continue
                
                # 计算中心点距离
                old_center = old_det.center
                new_center = new_det.center
                distance = np.sqrt((new_center[0] - old_center[0])**2 + 
                                 (new_center[1] - old_center[1])**2)
                
                if distance < self.match_distance and distance < best_match_distance:
                    best_match_idx = j
                    best_match_distance = distance
            
            if best_match_idx != -1:
                matched_old.append(i)
                matched_new.append(best_match_idx)
                
                new_det = new_detections[best_match_idx]
                
                # 判断是否在死区内
                if best_match_distance <= self.dead_zone:
                    comparison['dead_zone_unchanged'].append({
                        'old': old_det.to_dict(),
                        'new': new_det.to_dict(),
                        'distance': float(best_match_distance)
                    })
                else:
                    # 不在死区内，但不统计移动目标
                    # 根据目标类型处理：
                    # - 缺陷(Structual_Defect): 位置变化也视为未变化（因为缺陷不会移动）
                    # - 异物(Foreign_Object): 位置变化视为新增+消失
                    if "defect" in old_det.class_name.lower():
                        # 缺陷位置变化视为未变化
                        comparison['unchanged'].append(old_det.to_dict())
                    else:
                        # 异物位置变化视为移除旧目标，新增新目标
                        comparison['removed'].append(old_det.to_dict())
                        comparison['added'].append(new_det.to_dict())
                        # 标记为已处理，避免重复统计
                        comparison['moved_but_ignored'].append({
                            'old': old_det.to_dict(),
                            'new': new_det.to_dict(),
                            'distance': float(best_match_distance)
                        })
        
        # 识别新增的目标（未匹配的新目标）
        for j, new_det in enumerate(new_detections):
            if j not in matched_new:
                # 根据目标类型处理
                if "defect" in new_det.class_name.lower():
                    # 缺陷只会新增
                    comparison['added'].append(new_det.to_dict())
                else:
                    # 异物新增
                    comparison['added'].append(new_det.to_dict())
        
        # 识别消失的目标（未匹配的旧目标）
        for i, old_det in enumerate(old_detections):
            if i not in matched_old:
                # 根据目标类型处理
                if "defect" in old_det.class_name.lower():
                    # 缺陷不会消失，除非是误检
                    # 这里我们假设缺陷不会消失，所以不添加到removed
                    pass
                else:
                    # 异物消失
                    comparison['removed'].append(old_det.to_dict())
        
        # 更新按类别统计
        for det in comparison['added']:
            cls_name = det['class_name']
            if cls_name in comparison['by_class']:
                comparison['by_class'][cls_name]['added_count'] += 1
        
        for det in comparison['removed']:
            cls_name = det['class_name']
            if cls_name in comparison['by_class']:
                comparison['by_class'][cls_name]['removed_count'] += 1
        
        # 未变化目标（在死区内）
        comparison['unchanged'].extend([item['old'] for item in comparison['dead_zone_unchanged']])
        
        return comparison
    
    def save_report(self, comparison: Dict[str, Any], output_path: str):
        """保存对比报告"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("                   检测结果对比报告（优化版）\n")
            f.write("=" * 70 + "\n\n")
            
            # 总体统计
            f.write("一、总体统计\n")
            f.write("-" * 40 + "\n")
            f.write(f"旧图像检测总数: {comparison['summary']['old_total']}\n")
            f.write(f"新图像检测总数: {comparison['summary']['new_total']}\n")
            
            total_diff = comparison['summary']['total_difference']
            diff_sign = '+' if total_diff > 0 else ''
            f.write(f"总变化数量: {diff_sign}{total_diff}\n")
            
            f.write(f"新增目标: {len(comparison['added'])} 个\n")
            f.write(f"消失目标: {len(comparison['removed'])} 个\n")
            f.write(f"未变化目标: {len(comparison['unchanged'])} 个\n")
            f.write(f"死区内视为未变化: {len(comparison['dead_zone_unchanged'])} 个\n")
            f.write(f"移动目标（不统计）: {len(comparison['moved_but_ignored'])} 个\n\n")
            
            # 按类别统计
            f.write("二、按类别统计\n")
            f.write("-" * 40 + "\n")
            for cls_name, stats in comparison['by_class'].items():
                diff_sign = '+' if stats['difference'] > 0 else ''
                f.write(f"{cls_name}:\n")
                f.write(f"  旧图像: {stats['old_count']} 个\n")
                f.write(f"  新图像: {stats['new_count']} 个\n")
                f.write(f"  净变化: {diff_sign}{stats['difference']} 个\n")
                f.write(f"  新增数量: {stats['added_count']} 个\n")
                f.write(f"  消失数量: {stats['removed_count']} 个\n\n")
            
            # 新增目标详情
            if comparison['added']:
                f.write("三、新增目标详情\n")
                f.write("-" * 40 + "\n")
                for i, det in enumerate(comparison['added'], 1):
                    f.write(f"{i:3d}. 类别: {det['class_name']:20s} ")
                    f.write(f"位置: ({det['center'][0]:.1f}, {det['center'][1]:.1f}) ")
                    f.write(f"置信度: {det['confidence']:.3f} ")
                    f.write(f"面积: {det['area']:.1f}\n")
                f.write("\n")
            
            # 消失目标详情
            if comparison['removed']:
                f.write("四、消失目标详情\n")
                f.write("-" * 40 + "\n")
                for i, det in enumerate(comparison['removed'], 1):
                    f.write(f"{i:3d}. 类别: {det['class_name']:20s} ")
                    f.write(f"位置: ({det['center'][0]:.1f}, {det['center'][1]:.1f}) ")
                    f.write(f"置信度: {det['confidence']:.3f} ")
                    f.write(f"面积: {det['area']:.1f}\n")
                f.write("\n")
            
            # 死区内未变化目标详情
            if comparison['dead_zone_unchanged']:
                f.write("五、死区内未变化目标详情\n")
                f.write("-" * 40 + "\n")
                for i, item in enumerate(comparison['dead_zone_unchanged'], 1):
                    old_det = item['old']
                    f.write(f"{i:3d}. 类别: {old_det['class_name']:20s} ")
                    f.write(f"位置: ({old_det['center'][0]:.1f}, {old_det['center'][1]:.1f}) ")
                    f.write(f"移动距离: {item['distance']:.1f}像素\n")
                f.write("\n")
            
            f.write("=" * 70 + "\n")
            f.write(f"报告说明:\n")
            f.write(f"1. 死区范围: {self.dead_zone}像素，范围内位置波动视为未变化\n")
            f.write(f"2. 缺陷(Defect)目标不会消失，只会新增\n")
            f.write(f"3. 异物(Foreign Object)会新增和消失\n")
            f.write(f"4. 移动目标不单独统计，根据类型分别计入新增/消失\n")
            f.write("=" * 70 + "\n")
            f.write("报告生成时间: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write("=" * 70 + "\n")
        
        print(f"对比报告已保存到: {output_path}")

def draw_detections_with_info(image: np.ndarray, detections: List[Detection], 
                             color_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
                             thickness: int = 2,
                             show_confidence: bool = True,
                             show_class: bool = True) -> np.ndarray:
    """
    在图像上绘制检测框，带更多信息
    
    返回:
        np.ndarray: 绘制了检测框的图像
    """
    if color_map is None:
        color_map = {
            0: (0, 255, 0),    # 绿色 - 类别1
            1: (0, 0, 255),    # 红色 - 类别2
            2: (255, 0, 0),    # 蓝色 - 类别3
        }
    
    result = image.copy()
    
    for det in detections:
        # 边界框坐标
        x1, y1, x2, y2 = map(int, [det.x1, det.y1, det.x2, det.y2])
        
        # 颜色
        color = color_map.get(det.class_id, (255, 255, 0))
        
        # 绘制边界框
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
        
        # 构建标签文本
        label_parts = []
        if show_class:
            label_parts.append(det.class_name)
        if show_confidence:
            label_parts.append(f"{det.confidence:.2f}")
        if det.source != "window":
            label_parts.append(f"({det.source})")
        
        label = " ".join(label_parts)
        
        # 计算标签大小
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        (label_width, label_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness)
        
        # 标签背景
        cv2.rectangle(result, 
                     (x1, y1 - label_height - baseline - 2),
                     (x1 + label_width, y1),
                     color, -1)
        
        # 标签文字
        cv2.putText(result, label,
                   (x1, y1 - baseline - 2),
                   font, font_scale, (255, 255, 255), font_thickness)
    
    return result

def save_detections_json(detections: List[Detection], output_path: str):
    """保存检测结果到JSON文件"""
    detections_dict = [det.to_dict() for det in detections]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(detections_dict, f, ensure_ascii=False, indent=2)
    
    print(f"检测结果已保存到JSON文件: {output_path}")

def process_images_pipeline(
    old_image_path: str,
    new_image_path: str,
    model_path: str,
    template_path: str = "./moban/01_moban.bmp",
    output_dir: str = "./detection_results",
    window_size: int = 512,
    stride: int = 256,
    ignore_top: int = 700,
    enable_global_merge: bool = True,
    dead_zone: float = 20.0,
    save_images: bool = True,
    save_report: bool = True,
    save_json: bool = True
):
    """
    完整的图像处理与对比流程 - 优化版
    
    参数:
        dead_zone: 死区范围（像素），在此范围内的位置波动视为未变化
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    print("=" * 70)
    print("                  图像处理与检测对比系统（优化版）")
    print("=" * 70)
    
    # 1. 预处理图像
    print("\n1. 预处理图像...")
    
    # 预处理旧图像
    print(f"   处理旧图像: {old_image_path}")
    old_preprocessed, old_preprocessed_path = process_image_pipeline(
        input_image_path=old_image_path,
        template_path=template_path,
        output_dir=output_dir,
        output_suffix="_preprocessed"
    )
    
    if old_preprocessed is None:
        print("   旧图像预处理失败！")
        return
    
    # 预处理新图像
    print(f"   处理新图像: {new_image_path}")
    new_preprocessed, new_preprocessed_path = process_image_pipeline(
        input_image_path=new_image_path,
        template_path=template_path,
        output_dir=output_dir,
        output_suffix="_preprocessed"
    )
    
    if new_preprocessed is None:
        print("   新图像预处理失败！")
        return
    
    print(f"   预处理完成，图像尺寸: {old_preprocessed.shape[1]}x{old_preprocessed.shape[0]}")
    
    # 2. 初始化检测器
    print("\n2. 初始化YOLOv8检测器...")
    detector = SlidingWindowDetector(
        model_path=model_path,
        window_size=window_size,
        stride=stride,
        ignore_top=ignore_top,
        conf_threshold=0.25,
        iou_threshold=0.45
    )
    
    # 3. 检测旧图像
    print("\n3. 检测旧图像...")
    start_time = time.time()
    
    # 滑动窗口检测
    old_detections = detector.sliding_window_detection(old_preprocessed)
    
    # 全局合并重叠框
    if enable_global_merge and old_detections:
        print("   对旧图像进行全局重叠框合并...")
        old_detections = DetectionPostProcessor.merge_overlapping_detections(
            old_detections, 
            iou_threshold=0.3,
            max_aspect_ratio=3.0
        )
    
    old_detection_time = time.time() - start_time
    print(f"   旧图像检测完成，共检测到 {len(old_detections)} 个目标，耗时: {old_detection_time:.2f}秒")
    
    # 4. 检测新图像
    print("\n4. 检测新图像...")
    start_time = time.time()
    
    # 滑动窗口检测
    new_detections = detector.sliding_window_detection(new_preprocessed)
    
    # 全局合并重叠框
    if enable_global_merge and new_detections:
        print("   对新图像进行全局重叠框合并...")
        new_detections = DetectionPostProcessor.merge_overlapping_detections(
            new_detections, 
            iou_threshold=0.3,
            max_aspect_ratio=3.0
        )
    
    new_detection_time = time.time() - start_time
    print(f"   新图像检测完成，共检测到 {len(new_detections)} 个目标，耗时: {new_detection_time:.2f}秒")
    
    # 5. 比较结果
    print("\n5. 比较检测结果...")
    comparator = EnhancedDetectionComparator(
        detector.model.names, 
        dead_zone=dead_zone,
        match_distance=100.0
    )
    comparison = comparator.compare(old_detections, new_detections)
    
    # 6. 保存结果
    print("\n6. 保存结果...")
    
    # 保存JSON结果
    if save_json:
        old_json_path = os.path.join(output_dir, f"old_detections_{timestamp}.json")
        new_json_path = os.path.join(output_dir, f"new_detections_{timestamp}.json")
        save_detections_json(old_detections, old_json_path)
        save_detections_json(new_detections, new_json_path)
    
    # 保存检测结果图像
    if save_images and old_preprocessed is not None and new_preprocessed is not None:
        print("   保存带检测框的图像...")
        
        # 绘制检测框（带更多信息）
        old_detected = draw_detections_with_info(old_preprocessed, old_detections)
        new_detected = draw_detections_with_info(new_preprocessed, new_detections)
        
        # 保存图像
        old_output_path = os.path.join(output_dir, f"old_detected_{timestamp}.bmp")
        new_output_path = os.path.join(output_dir, f"new_detected_{timestamp}.bmp")
        
        cv2.imwrite(old_output_path, old_detected)
        cv2.imwrite(new_output_path, new_detected)
        
        print(f"   旧图像检测结果: {old_output_path}")
        print(f"   新图像检测结果: {new_output_path}")
        
        # 保存对比图像（并排显示）
        if old_detected.shape == new_detected.shape:
            # 添加文字说明
            cv2.putText(old_detected, "旧图像", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            cv2.putText(new_detected, "新图像", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
            
            comparison_image = np.hstack([old_detected, new_detected])
            comparison_path = os.path.join(output_dir, f"comparison_{timestamp}.bmp")
            cv2.imwrite(comparison_path, comparison_image)
            print(f"   对比图像: {comparison_path}")
    
    # 保存对比报告
    if save_report:
        report_path = os.path.join(output_dir, f"detection_comparison_{timestamp}.txt")
        comparator.save_report(comparison, report_path)
    
    # 7. 打印摘要
    print("\n" + "=" * 70)
    print("处理完成！")
    print("=" * 70)
    print(f"处理时间: {old_detection_time + new_detection_time:.2f}秒")
    print(f"死区范围: {dead_zone}像素")
    print(f"旧图像目标数: {len(old_detections)}")
    print(f"新图像目标数: {len(new_detections)}")
    
    # 按类别打印统计
    print("\n按类别统计:")
    for cls_name, stats in comparison['by_class'].items():
        diff_sign = '+' if stats['difference'] > 0 else ''
        print(f"  {cls_name}:")
        print(f"    旧: {stats['old_count']}个, 新: {stats['new_count']}个")
        print(f"    净变化: {diff_sign}{stats['difference']}个")
        print(f"    新增: {stats['added_count']}个, 消失: {stats['removed_count']}个")
    
    print(f"\n新增目标: {len(comparison['added'])}个")
    print(f"消失目标: {len(comparison['removed'])}个")
    print(f"未变化目标: {len(comparison['unchanged'])}个")
    print(f"死区内未变化: {len(comparison['dead_zone_unchanged'])}个")
    
    print(f"\n所有结果已保存到目录: {output_dir}")
    print("=" * 70)

def main():
    """主函数：解析命令行参数"""
    
    parser = argparse.ArgumentParser(
        description="基于YOLOv8的图像检测与对比系统（优化版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python detection_enhanced.py --old ./old.bmp --new ./new.bmp --model ./best.pt
  python detection_enhanced.py -o ./old.bmp -n ./new.bmp -m ./best.pt --dead-zone 20
        """
    )
    
    # 必需参数
    parser.add_argument("--old", "-o", type=str, required=True,
                       help="旧图像路径")
    parser.add_argument("--new", "-n", type=str, required=True,
                       help="新图像路径")
    parser.add_argument("--model", "-m", type=str, required=True,
                       help="YOLOv8模型路径 (best.pt)")
    
    # 可选参数
    parser.add_argument("--template", "-t", type=str, 
                       default="./moban/01_moban.bmp",
                       help="预处理模板路径 (默认: ./moban/01_moban.bmp)")
    parser.add_argument("--output", "-out", type=str,
                       default="./detection_results",
                       help="输出目录 (默认: ./detection_results)")
    parser.add_argument("--window", "-w", type=int,
                       default=512,
                       help="滑动窗口大小 (默认: 512)")
    parser.add_argument("--stride", "-s", type=int,
                       default=256,
                       help="滑动步长 (默认: 256，50%重叠)")
    parser.add_argument("--ignore-top", "-it", type=int,
                       default=700,
                       help="忽略顶部像素数 (默认: 700)")
    parser.add_argument("--dead-zone", "-dz", type=float,
                       default=20.0,
                       help="死区范围（像素），范围内位置波动视为未变化 (默认: 20)")
    parser.add_argument("--no-merge", action="store_true",
                       help="禁用全局重叠框合并")
    parser.add_argument("--no-images", action="store_true",
                       help="不保存检测结果图像")
    parser.add_argument("--no-report", action="store_true",
                       help="不保存对比报告")
    parser.add_argument("--no-json", action="store_true",
                       help="不保存JSON格式的检测结果")
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    for path in [args.old, args.new, args.model]:
        if not os.path.exists(path):
            print(f"错误: 文件不存在 - {path}")
            return
    
    # 运行处理流程
    try:
        process_images_pipeline(
            old_image_path=args.old,
            new_image_path=args.new,
            model_path=args.model,
            template_path=args.template,
            output_dir=args.output,
            window_size=args.window,
            stride=args.stride,
            ignore_top=args.ignore_top,
            enable_global_merge=not args.no_merge,
            dead_zone=args.dead_zone,
            save_images=not args.no_images,
            save_report=not args.no_report,
            save_json=not args.no_json
        )
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()