"""
基于YOLOv8的目标检测与对比系统
功能：预处理 → 滑动窗口检测 → 多尺度重检测 → 结果对比
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
    
    @property
    def center(self) -> Tuple[float, float]:
        """返回边界框中心点"""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
    
    @property
    def area(self) -> float:
        """返回边界框面积"""
        return (self.x2 - self.x1) * (self.y2 - self.y1)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'bbox': [self.x1, self.y1, self.x2, self.y2],
            'confidence': self.confidence,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'center': self.center,
            'area': self.area
        }

class SlidingWindowDetector:
    """滑动窗口检测器，处理大图像"""
    
    def __init__(self, model_path: str, window_size: int = 512, 
                 stride: int = 256, ignore_top: int = 700,
                 conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        初始化检测器
        
        参数:
            model_path: YOLOv8模型路径
            window_size: 滑动窗口大小
            stride: 滑动步长
            ignore_top: 忽略顶部像素数
            conf_threshold: 置信度阈值
            iou_threshold: NMS IoU阈值
        """
        self.model = YOLO(model_path)
        self.window_size = window_size
        self.stride = stride
        self.ignore_top = ignore_top
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")
        
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
        
        # 滑动窗口
        for y in range(start_y, height - self.window_size + 1, self.stride):
            for x in range(0, width - self.window_size + 1, self.stride):
                window_count += 1
                if window_count % 100 == 0:
                    print(f"处理窗口中... {window_count}/{total_windows}")
                
                # 提取窗口
                window = image[y:y+self.window_size, x:x+self.window_size]
                
                # YOLOv8检测
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
                                class_name=self.model.names[cls_id]
                            )
                            all_detections.append(detection)
        
        print(f"滑动窗口检测完成，初步检测到 {len(all_detections)} 个目标")
        return all_detections
    
    def non_max_suppression(self, detections: List[Detection], 
                          iou_threshold: float = 0.5) -> List[Detection]:
        """
        非极大值抑制 (NMS)
        
        返回:
            List[Detection]: 抑制后的检测结果
        """
        if not detections:
            return []
        
        # 按置信度排序
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        keep = []
        suppressed = set()
        
        for i in range(len(detections)):
            if i in suppressed:
                continue
            
            keep.append(detections[i])
            
            for j in range(i + 1, len(detections)):
                if j in suppressed:
                    continue
                
                iou = self._calculate_iou(detections[i], detections[j])
                if iou > iou_threshold:
                    suppressed.add(j)
        
        print(f"NMS后保留 {len(keep)} 个目标 (原始: {len(detections)})")
        return keep
    
    def cluster_detections(self, detections: List[Detection], 
                          distance_threshold: float = 100.0) -> List[List[Detection]]:
        """
        对检测结果进行聚类，用于后续精细检测
        
        返回:
            List[List[Detection]]: 聚类列表
        """
        if not detections:
            return []
        
        clusters = []
        assigned = set()
        
        for i, det in enumerate(detections):
            if i in assigned:
                continue
            
            # 创建新聚类
            cluster = [det]
            assigned.add(i)
            
            # 寻找邻近的检测框
            for j, other_det in enumerate(detections):
                if j in assigned:
                    continue
                
                # 计算中心点距离
                center1 = det.center
                center2 = other_det.center
                distance = np.sqrt((center2[0] - center1[0])**2 + 
                                 (center2[1] - center1[1])**2)
                
                # 检查是否同一类别且距离相近
                if det.class_id == other_det.class_id and distance < distance_threshold:
                    cluster.append(other_det)
                    assigned.add(j)
            
            # 只保留有多个检测框的聚类
            if len(cluster) > 1:
                clusters.append(cluster)
        
        print(f"找到 {len(clusters)} 个聚类区域需要精细检测")
        return clusters
    
    def refine_cluster_detections(self, image: np.ndarray, 
                                clusters: List[List[Detection]]) -> List[Detection]:
        """
        对聚类区域进行精细检测
        
        返回:
            List[Detection]: 精细检测后的结果
        """
        refined_detections = []
        
        for cluster_idx, cluster in enumerate(clusters):
            if len(cluster) == 0:
                continue
            
            # 计算聚类区域边界
            x_min = min(det.x1 for det in cluster)
            y_min = min(det.y1 for det in cluster)
            x_max = max(det.x2 for det in cluster)
            y_max = max(det.y2 for det in cluster)
            
            # 扩展区域（扩大50%）
            width = x_max - x_min
            height = y_max - y_min
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            
            expanded_width = width * 1.5
            expanded_height = height * 1.5
            
            # 计算扩展后的边界（确保为整数）
            x1 = int(max(0, center_x - expanded_width / 2))
            y1 = int(max(0, center_y - expanded_height / 2))
            x2 = int(min(image.shape[1], center_x + expanded_width / 2))
            y2 = int(min(image.shape[0], center_y + expanded_height / 2))
            
            # 确保区域有效
            if x2 <= x1 or y2 <= y1:
                continue
            
            # 裁剪区域
            crop = image[y1:y2, x1:x2]
            
            if crop.size == 0:
                continue
            
            # 调整到512x512进行检测
            crop_resized = cv2.resize(crop, (512, 512))
            
            # 在调整后的区域进行检测
            results = self.model(crop_resized, conf=self.conf_threshold * 0.8, 
                               iou=self.iou_threshold, device=self.device, verbose=False)
            
            if results[0].boxes is None or len(results[0].boxes) == 0:
                continue
            
            # 获取检测结果
            boxes = results[0].boxes
            best_idx = torch.argmax(boxes.conf).item()
            best_box = boxes.xyxy[best_idx].cpu().numpy()
            best_conf = boxes.conf[best_idx].cpu().numpy()
            best_cls = int(boxes.cls[best_idx].cpu().numpy())
            
            # 计算缩放比例
            scale_x = (x2 - x1) / 512
            scale_y = (y2 - y1) / 512
            
            # 转换回原图坐标
            x1_global = x1 + best_box[0] * scale_x
            y1_global = y1 + best_box[1] * scale_y
            x2_global = x1 + best_box[2] * scale_x
            y2_global = y1 + best_box[3] * scale_y
            
            refined_detection = Detection(
                x1=float(x1_global),
                y1=float(y1_global),
                x2=float(x2_global),
                y2=float(y2_global),
                confidence=float(best_conf),
                class_id=best_cls,
                class_name=self.model.names[best_cls]
            )
            refined_detections.append(refined_detection)
        
        print(f"精细检测完成，得到 {len(refined_detections)} 个优化结果")
        return refined_detections
    
    @staticmethod
    def _calculate_iou(det1: Detection, det2: Detection) -> float:
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

class DetectionComparator:
    """检测结果比较器"""
    
    def __init__(self, class_names: Dict[int, str], distance_threshold: float = 50.0):
        self.class_names = class_names
        self.distance_threshold = distance_threshold
        
    def compare(self, old_detections: List[Detection], 
                new_detections: List[Detection]) -> Dict[str, Any]:
        """
        比较新旧检测结果
        
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
            'moved': [],
            'unchanged': []
        }
        
        # 按类别统计
        for cls_id, cls_name in self.class_names.items():
            old_cls = [d for d in old_detections if d.class_id == cls_id]
            new_cls = [d for d in new_detections if d.class_id == cls_id]
            
            comparison['by_class'][cls_name] = {
                'old_count': len(old_cls),
                'new_count': len(new_cls),
                'difference': len(new_cls) - len(old_cls)
            }
        
        # 匹配检测结果
        matched_old = []
        matched_new = []
        matches = []
        
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
                
                if distance < self.distance_threshold and distance < best_match_distance:
                    best_match_idx = j
                    best_match_distance = distance
            
            if best_match_idx != -1:
                matched_old.append(i)
                matched_new.append(best_match_idx)
                
                new_det = new_detections[best_match_idx]
                old_center = old_det.center
                new_center = new_det.center
                distance = np.sqrt((new_center[0] - old_center[0])**2 + 
                                 (new_center[1] - old_center[1])**2)
                
                if distance > 10.0:  # 移动阈值
                    comparison['moved'].append({
                        'old': old_det.to_dict(),
                        'new': new_det.to_dict(),
                        'distance': float(distance)
                    })
                else:
                    comparison['unchanged'].append(old_det.to_dict())
        
        # 识别新增的目标
        for j, new_det in enumerate(new_detections):
            if j not in matched_new:
                comparison['added'].append(new_det.to_dict())
        
        # 识别消失的目标
        for i, old_det in enumerate(old_detections):
            if i not in matched_old:
                comparison['removed'].append(old_det.to_dict())
        
        return comparison
    
    def save_report(self, comparison: Dict[str, Any], output_path: str):
        """保存对比报告"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("                   检测结果对比报告\n")
            f.write("=" * 70 + "\n\n")
            
            # 总体统计
            f.write("一、总体统计\n")
            f.write("-" * 40 + "\n")
            f.write(f"旧图像检测总数: {comparison['summary']['old_total']}\n")
            f.write(f"新图像检测总数: {comparison['summary']['new_total']}\n")
            f.write(f"总变化数量: {comparison['summary']['total_difference']:+\n}")
            f.write(f"新增目标: {len(comparison['added'])} 个\n")
            f.write(f"消失目标: {len(comparison['removed'])} 个\n")
            f.write(f"移动目标: {len(comparison['moved'])} 个\n")
            f.write(f"未变化目标: {len(comparison['unchanged'])} 个\n\n")
            
            # 按类别统计
            f.write("二、按类别统计\n")
            f.write("-" * 40 + "\n")
            for cls_name, stats in comparison['by_class'].items():
                diff_sign = '+' if stats['difference'] > 0 else ''
                f.write(f"{cls_name}:\n")
                f.write(f"  旧图像: {stats['old_count']} 个\n")
                f.write(f"  新图像: {stats['new_count']} 个\n")
                f.write(f"  变化: {diff_sign}{stats['difference']} 个\n\n")
            
            # 详细变化
            if comparison['added']:
                f.write("三、新增目标详情\n")
                f.write("-" * 40 + "\n")
                for i, det in enumerate(comparison['added'], 1):
                    f.write(f"{i:3d}. 类别: {det['class_name']:10s} ")
                    f.write(f"位置: ({det['center'][0]:.1f}, {det['center'][1]:.1f}) ")
                    f.write(f"置信度: {det['confidence']:.3f}\n")
                f.write("\n")
            
            if comparison['removed']:
                f.write("四、消失目标详情\n")
                f.write("-" * 40 + "\n")
                for i, det in enumerate(comparison['removed'], 1):
                    f.write(f"{i:3d}. 类别: {det['class_name']:10s} ")
                    f.write(f"位置: ({det['center'][0]:.1f}, {det['center'][1]:.1f}) ")
                    f.write(f"置信度: {det['confidence']:.3f}\n")
                f.write("\n")
            
            if comparison['moved']:
                f.write("五、移动目标详情\n")
                f.write("-" * 40 + "\n")
                for i, move in enumerate(comparison['moved'], 1):
                    old_det = move['old']
                    new_det = move['new']
                    f.write(f"{i:3d}. 类别: {old_det['class_name']:10s}\n")
                    f.write(f"     旧位置: ({old_det['center'][0]:.1f}, {old_det['center'][1]:.1f})\n")
                    f.write(f"     新位置: ({new_det['center'][0]:.1f}, {new_det['center'][1]:.1f})\n")
                    f.write(f"     移动距离: {move['distance']:.1f} 像素\n\n")
            
            f.write("=" * 70 + "\n")
            f.write("报告生成时间: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
            f.write("=" * 70 + "\n")
        
        print(f"对比报告已保存到: {output_path}")

def draw_detections(image: np.ndarray, detections: List[Detection], 
                   color_map: Optional[Dict[int, Tuple[int, int, int]]] = None,
                   thickness: int = 2) -> np.ndarray:
    """
    在图像上绘制检测框
    
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
        
        # 绘制标签
        label = f"{det.class_name}: {det.confidence:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # 标签背景
        cv2.rectangle(result, 
                     (x1, y1 - label_height - baseline),
                     (x1 + label_width, y1),
                     color, -1)
        
        # 标签文字
        cv2.putText(result, label,
                   (x1, y1 - baseline),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
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
    enable_refinement: bool = True,
    save_images: bool = True,
    save_report: bool = True,
    save_json: bool = True
):
    """
    完整的图像处理与对比流程
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    print("=" * 70)
    print("                  图像处理与检测对比系统")
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
    
    # 非极大值抑制
    if old_detections:
        old_detections = detector.non_max_suppression(old_detections, iou_threshold=0.5)
    
    # 精细检测（如果启用）
    if enable_refinement and old_detections:
        print("   对旧图像进行精细检测...")
        clusters = detector.cluster_detections(old_detections, distance_threshold=100.0)
        if clusters:
            refined_detections = detector.refine_cluster_detections(old_preprocessed, clusters)
            # 合并结果
            old_detections.extend(refined_detections)
            old_detections = detector.non_max_suppression(old_detections, iou_threshold=0.5)
    
    old_detection_time = time.time() - start_time
    print(f"   旧图像检测完成，共检测到 {len(old_detections)} 个目标，耗时: {old_detection_time:.2f}秒")
    
    # 4. 检测新图像
    print("\n4. 检测新图像...")
    start_time = time.time()
    
    # 滑动窗口检测
    new_detections = detector.sliding_window_detection(new_preprocessed)
    
    # 非极大值抑制
    if new_detections:
        new_detections = detector.non_max_suppression(new_detections, iou_threshold=0.5)
    
    # 精细检测（如果启用）
    if enable_refinement and new_detections:
        print("   对新图像进行精细检测...")
        clusters = detector.cluster_detections(new_detections, distance_threshold=100.0)
        if clusters:
            refined_detections = detector.refine_cluster_detections(new_preprocessed, clusters)
            # 合并结果
            new_detections.extend(refined_detections)
            new_detections = detector.non_max_suppression(new_detections, iou_threshold=0.5)
    
    new_detection_time = time.time() - start_time
    print(f"   新图像检测完成，共检测到 {len(new_detections)} 个目标，耗时: {new_detection_time:.2f}秒")
    
    # 5. 比较结果
    print("\n5. 比较检测结果...")
    comparator = DetectionComparator(detector.model.names, distance_threshold=50.0)
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
    if save_images:
        print("   保存带检测框的图像...")
        
        # 绘制检测框
        old_detected = draw_detections(old_preprocessed, old_detections)
        new_detected = draw_detections(new_preprocessed, new_detections)
        
        # 保存图像
        old_output_path = os.path.join(output_dir, f"old_detected_{timestamp}.bmp")
        new_output_path = os.path.join(output_dir, f"new_detected_{timestamp}.bmp")
        
        cv2.imwrite(old_output_path, old_detected)
        cv2.imwrite(new_output_path, new_detected)
        
        print(f"   旧图像检测结果: {old_output_path}")
        print(f"   新图像检测结果: {new_output_path}")
        
        # 保存对比图像（并排显示）
        if old_detected.shape == new_detected.shape:
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
    print(f"旧图像目标数: {len(old_detections)}")
    print(f"新图像目标数: {len(new_detections)}")
    print(f"新增目标: {len(comparison['added'])}")
    print(f"消失目标: {len(comparison['removed'])}")
    print(f"移动目标: {len(comparison['moved'])}")
    print(f"未变化目标: {len(comparison['unchanged'])}")
    
    print("\n按类别统计:")
    for cls_name, stats in comparison['by_class'].items():
        diff_sign = '+' if stats['difference'] > 0 else ''
        print(f"  {cls_name}: 旧{stats['old_count']} → 新{stats['new_count']} ({diff_sign}{stats['difference']})")
    
    print(f"\n所有结果已保存到目录: {output_dir}")
    print("=" * 70)

def main():
    """主函数：解析命令行参数"""
    
    parser = argparse.ArgumentParser(
        description="基于YOLOv8的图像检测与对比系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python detection_pipeline.py --old ./old.bmp --new ./new.bmp --model ./best.pt
  python detection_pipeline.py -o ./old.bmp -n ./new.bmp -m ./best.pt -out ./results
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
                       help="滑动步长 (默认: 256，50%%重叠)")
    parser.add_argument("--ignore-top", "-it", type=int,
                       default=700,
                       help="忽略顶部像素数 (默认: 700)")
    parser.add_argument("--no-refine", action="store_true",
                       help="禁用精细检测")
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
            enable_refinement=not args.no_refine,
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