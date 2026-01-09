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
from typing import List, Tuple, Dict, Any
import time

class SlidingWindowDetector:
    """滑动窗口检测器，处理大图像"""
    
    def __init__(self, model_path: str, window_size: int = 512, 
                 stride: int = 256, ignore_top: int = 700):
        """
        初始化检测器
        
        参数:
            model_path: YOLOv8模型路径
            window_size: 滑动窗口大小
            stride: 滑动步长
            ignore_top: 忽略顶部像素数
        """
        self.model = YOLO(model_path)
        self.window_size = window_size
        self.stride = stride
        self.ignore_top = ignore_top
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def sliding_window_detection(self, image: np.ndarray, 
                               conf_threshold: float = 0.25,
                               iou_threshold: float = 0.45) -> List[Dict]:
        """
        滑动窗口检测
        
        返回:
            List[Dict]: 每个检测结果包含bbox, confidence, class_id
        """
        height, width = image.shape[:2]
        all_detections = []
        
        # 忽略顶部区域
        start_y = self.ignore_top
        
        # 计算滑动窗口位置
        for y in range(start_y, height - self.window_size + 1, self.stride):
            for x in range(0, width - self.window_size + 1, self.stride):
                # 提取窗口
                window = image[y:y+self.window_size, x:x+self.window_size]
                
                # YOLOv8检测
                results = self.model(window, conf=conf_threshold, iou=iou_threshold, device=self.device)
                
                # 处理检测结果
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
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
                            
                            detection = {
                                'bbox': [x1_global, y1_global, x2_global, y2_global],
                                'confidence': float(conf),
                                'class_id': cls_id,
                                'class_name': self.model.names[cls_id]
                            }
                            all_detections.append(detection)
        
        return all_detections
    
    def cluster_detections(self, detections: List[Dict], 
                          iou_threshold: float = 0.5) -> List[Dict]:
        """
        聚类检测结果，找到重叠区域
        
        返回:
            List[Dict]: 聚类中心信息
        """
        if not detections:
            return []
        
        # 按置信度排序
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        clusters = []
        used_indices = set()
        
        for i, det in enumerate(detections):
            if i in used_indices:
                continue
                
            # 创建新聚类
            cluster = {
                'center': self._get_bbox_center(det['bbox']),
                'bboxes': [det],
                'avg_confidence': det['confidence']
            }
            
            used_indices.add(i)
            
            # 寻找重叠的检测框
            for j, other_det in enumerate(detections):
                if j in used_indices:
                    continue
                    
                iou = self._calculate_iou(det['bbox'], other_det['bbox'])
                if iou > iou_threshold:
                    cluster['bboxes'].append(other_det)
                    # 更新聚类中心
                    cluster['center'] = self._calculate_cluster_center(cluster['bboxes'])
                    cluster['avg_confidence'] = np.mean([d['confidence'] for d in cluster['bboxes']])
                    used_indices.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def refine_detection(self, image: np.ndarray, cluster: Dict,
                        scale_factor: float = 2.0) -> Dict:
        """
        对聚类区域进行精细检测
        
        参数:
            image: 原图像
            cluster: 聚类信息
            scale_factor: 缩放因子，扩大检测区域
        
        返回:
            Dict: 优化后的检测结果
        """
        center_x, center_y = cluster['center']
        
        # 计算扩展后的区域
        expanded_size = int(self.window_size * scale_factor)
        half_size = expanded_size // 2
        
        # 计算裁剪边界
        x1 = max(0, center_x - half_size)
        y1 = max(0, center_y - half_size)
        x2 = min(image.shape[1], center_x + half_size)
        y2 = min(image.shape[0], center_y + half_size)
        
        # 裁剪区域
        crop = image[y1:y2, x1:x2]
        
        if crop.size == 0:
            return None
        
        # 调整大小进行检测
        results = self.model(crop, conf=0.2, iou=0.4, device=self.device)
        
        if results[0].boxes is None:
            return None
        
        # 选择置信度最高的检测
        boxes = results[0].boxes
        best_idx = torch.argmax(boxes.conf).item()
        best_box = boxes.xyxy[best_idx].cpu().numpy()
        best_conf = boxes.conf[best_idx].cpu().numpy()
        best_cls = int(boxes.cls[best_idx].cpu().numpy())
        
        # 转换回原图坐标
        x1_global = x1 + best_box[0]
        y1_global = y1 + best_box[1]
        x2_global = x1 + best_box[2]
        y2_global = y1 + best_box[3]
        
        refined_detection = {
            'bbox': [x1_global, y1_global, x2_global, y2_global],
            'confidence': float(best_conf),
            'class_id': best_cls,
            'class_name': self.model.names[best_cls],
            'refined': True
        }
        
        return refined_detection
    
    @staticmethod
    def _get_bbox_center(bbox: List[float]) -> Tuple[float, float]:
        """计算边界框中心点"""
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    @staticmethod
    def _calculate_iou(bbox1: List[float], bbox2: List[float]) -> float:
        """计算IoU"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # 计算交集
        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)
        
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        
        # 计算并集
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    @staticmethod
    def _calculate_cluster_center(bboxes: List[Dict]) -> Tuple[float, float]:
        """计算聚类中心"""
        centers = []
        for det in bboxes:
            x1, y1, x2, y2 = det['bbox']
            centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
        
        avg_x = np.mean([c[0] for c in centers])
        avg_y = np.mean([c[1] for c in centers])
        
        return (avg_x, avg_y)

class DetectionComparator:
    """检测结果比较器"""
    
    def __init__(self, class_names: Dict):
        self.class_names = class_names
        
    def compare_detections(self, old_detections: List[Dict], 
                          new_detections: List[Dict],
                          distance_threshold: float = 50.0) -> Dict:
        """
        比较新旧检测结果
        
        返回:
            Dict: 包含对比结果的字典
        """
        comparison = {
            'old_count': len(old_detections),
            'new_count': len(new_detections),
            'added': [],
            'removed': [],
            'moved': [],
            'unchanged': [],
            'class_summary': {}
        }
        
        # 按类别统计
        for cls_id, cls_name in self.class_names.items():
            old_cls = [d for d in old_detections if d['class_id'] == cls_id]
            new_cls = [d for d in new_detections if d['class_id'] == cls_id]
            
            comparison['class_summary'][cls_name] = {
                'old': len(old_cls),
                'new': len(new_cls),
                'difference': len(new_cls) - len(old_cls)
            }
        
        # 寻找匹配的目标
        matched_old = set()
        matched_new = set()
        
        for i, old_det in enumerate(old_detections):
            for j, new_det in enumerate(new_detections):
                if j in matched_new:
                    continue
                    
                # 检查是否同一类别且位置相近
                if old_det['class_id'] != new_det['class_id']:
                    continue
                    
                # 计算中心点距离
                old_center = self._get_detection_center(old_det)
                new_center = self._get_detection_center(new_det)
                distance = np.sqrt((new_center[0] - old_center[0])**2 + 
                                  (new_center[1] - old_center[1])**2)
                
                if distance < distance_threshold:
                    matched_old.add(i)
                    matched_new.add(j)
                    
                    # 检查是否移动
                    if distance > 10.0:
                        comparison['moved'].append({
                            'old': old_det,
                            'new': new_det,
                            'distance': distance
                        })
                    else:
                        comparison['unchanged'].append(old_det)
                    break
        
        # 识别新增和消失的目标
        for i, old_det in enumerate(old_detections):
            if i not in matched_old:
                comparison['removed'].append(old_det)
        
        for j, new_det in enumerate(new_detections):
            if j not in matched_new:
                comparison['added'].append(new_det)
        
        return comparison
    
    @staticmethod
    def _get_detection_center(detection: Dict) -> Tuple[float, float]:
        """获取检测目标中心点"""
        x1, y1, x2, y2 = detection['bbox']
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def save_comparison_report(self, comparison: Dict, output_path: str):
        """保存对比报告到txt文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("检测结果对比报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("一、总体统计\n")
            f.write(f"旧图像检测总数: {comparison['old_count']}\n")
            f.write(f"新图像检测总数: {comparison['new_count']}\n")
            f.write(f"目标变化总数: {len(comparison['added']) + len(comparison['removed'])}\n\n")
            
            f.write("二、按类别统计\n")
            for cls_name, stats in comparison['class_summary'].items():
                f.write(f"{cls_name}:\n")
                f.write(f"  旧图像: {stats['old']}个\n")
                f.write(f"  新图像: {stats['new']}个\n")
                f.write(f"  变化量: {stats['difference']:+d}个\n\n")
            
            f.write("三、详细变化\n")
            f.write(f"新增目标: {len(comparison['added'])}个\n")
            for i, det in enumerate(comparison['added'], 1):
                f.write(f"  {i}. 类别: {det['class_name']}, "
                       f"位置: ({det['bbox'][0]:.1f}, {det['bbox'][1]:.1f}), "
                       f"置信度: {det['confidence']:.3f}\n")
            
            f.write(f"\n消失目标: {len(comparison['removed'])}个\n")
            for i, det in enumerate(comparison['removed'], 1):
                f.write(f"  {i}. 类别: {det['class_name']}, "
                       f"位置: ({det['bbox'][0]:.1f}, {det['bbox'][1]:.1f}), "
                       f"置信度: {det['confidence']:.3f}\n")
            
            f.write(f"\n移动目标: {len(comparison['moved'])}个\n")
            for i, move in enumerate(comparison['moved'], 1):
                old_det = move['old']
                new_det = move['new']
                f.write(f"  {i}. 类别: {old_det['class_name']}\n")
                f.write(f"     旧位置: ({old_det['bbox'][0]:.1f}, {old_det['bbox'][1]:.1f})\n")
                f.write(f"     新位置: ({new_det['bbox'][0]:.1f}, {new_det['bbox'][1]:.1f})\n")
                f.write(f"     移动距离: {move['distance']:.1f}像素\n")

def draw_detections(image: np.ndarray, detections: List[Dict], 
                   color_map: Dict[int, Tuple] = None) -> np.ndarray:
    """
    在图像上绘制检测框
    
    参数:
        image: 输入图像
        detections: 检测结果列表
        color_map: 类别颜色映射
    
    返回:
        np.ndarray: 绘制了检测框的图像
    """
    if color_map is None:
        color_map = {
            0: (0, 255, 0),  # 绿色
            1: (0, 0, 255),  # 红色
            2: (255, 0, 0),  # 蓝色
        }
    
    result = image.copy()
    
    for det in detections:
        bbox = det['bbox']
        cls_id = det['class_id']
        conf = det['confidence']
        cls_name = det.get('class_name', f'Class_{cls_id}')
        
        color = color_map.get(cls_id, (255, 255, 0))
        
        # 绘制边界框
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签
        label = f"{cls_name}: {conf:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        cv2.rectangle(result, 
                     (x1, y1 - label_height - baseline),
                     (x1 + label_width, y1),
                     color, -1)
        cv2.putText(result, label,
                   (x1, y1 - baseline),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return result

def process_images_pipeline(
    old_image_path: str,
    new_image_path: str,
    model_path: str,
    template_path: str = "./moban/01_moban.bmp",
    output_dir: str = "./detection_results",
    window_size: int = 512,
    stride: int = 256,
    ignore_top: int = 700,
    refine_clusters: bool = True,
    save_images: bool = True,
    save_report: bool = True
):
    """
    完整的图像处理与对比流程
    
    参数:
        old_image_path: 旧图像路径
        new_image_path: 新图像路径
        model_path: YOLOv8模型路径
        template_path: 预处理模板路径
        output_dir: 输出目录
        window_size: 滑动窗口大小
        stride: 滑动步长
        ignore_top: 忽略顶部像素
        refine_clusters: 是否进行精细检测
        save_images: 是否保存带检测框的图像
        save_report: 是否保存对比报告
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("开始图像处理与检测流程")
    print("=" * 60)
    
    # 1. 预处理图像
    print("\n1. 预处理图像...")
    print("   处理旧图像...")
    old_img, old_preprocessed_path = process_image_pipeline(
        input_image_path=old_image_path,
        template_path=template_path,
        output_dir=output_dir,
        output_suffix="_preprocessed"
    )
    
    print("   处理新图像...")
    new_img, new_preprocessed_path = process_image_pipeline(
        input_image_path=new_image_path,
        template_path=template_path,
        output_dir=output_dir,
        output_suffix="_preprocessed"
    )
    
    if old_img is None or new_img is None:
        print("预处理失败！")
        return
    
    # 2. 加载检测器
    print("\n2. 加载YOLOv8模型...")
    detector = SlidingWindowDetector(
        model_path=model_path,
        window_size=window_size,
        stride=stride,
        ignore_top=ignore_top
    )
    
    # 3. 检测旧图像
    print("\n3. 检测旧图像...")
    start_time = time.time()
    old_detections = detector.sliding_window_detection(old_img)
    
    if refine_clusters and old_detections:
        print("   对旧图像进行精细检测...")
        clusters = detector.cluster_detections(old_detections)
        refined_detections = []
        
        for cluster in clusters:
            refined = detector.refine_detection(old_img, cluster)
            if refined:
                refined_detections.append(refined)
        
        old_detections = refined_detections
    
    print(f"   检测完成，发现 {len(old_detections)} 个目标，耗时: {time.time()-start_time:.2f}秒")
    
    # 4. 检测新图像
    print("\n4. 检测新图像...")
    start_time = time.time()
    new_detections = detector.sliding_window_detection(new_img)
    
    if refine_clusters and new_detections:
        print("   对新图像进行精细检测...")
        clusters = detector.cluster_detections(new_detections)
        refined_detections = []
        
        for cluster in clusters:
            refined = detector.refine_detection(new_img, cluster)
            if refined:
                refined_detections.append(refined)
        
        new_detections = refined_detections
    
    print(f"   检测完成，发现 {len(new_detections)} 个目标，耗时: {time.time()-start_time:.2f}秒")
    
    # 5. 比较结果
    print("\n5. 比较检测结果...")
    comparator = DetectionComparator(detector.model.names)
    comparison = comparator.compare_detections(old_detections, new_detections)
    
    # 6. 保存结果
    print("\n6. 保存结果...")
    
    # 保存检测结果图像
    if save_images:
        print("   保存带检测框的图像...")
        old_detected = draw_detections(old_img, old_detections)
        new_detected = draw_detections(new_img, new_detections)
        
        old_output_path = os.path.join(output_dir, "old_detected.bmp")
        new_output_path = os.path.join(output_dir, "new_detected.bmp")
        
        cv2.imwrite(old_output_path, old_detected)
        cv2.imwrite(new_output_path, new_detected)
        
        print(f"   旧图像检测结果保存到: {old_output_path}")
        print(f"   新图像检测结果保存到: {new_output_path}")
    
    # 保存对比报告
    if save_report:
        report_path = os.path.join(output_dir, "detection_comparison.txt")
        comparator.save_comparison_report(comparison, report_path)
        print(f"   对比报告保存到: {report_path}")
    
    # 7. 打印摘要
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"旧图像检测总数: {comparison['old_count']}")
    print(f"新图像检测总数: {comparison['new_count']}")
    print(f"新增目标: {len(comparison['added'])} 个")
    print(f"消失目标: {len(comparison['removed'])} 个")
    print(f"移动目标: {len(comparison['moved'])} 个")
    
    # 按类别显示统计
    print("\n按类别统计:")
    for cls_name, stats in comparison['class_summary'].items():
        diff_sign = '+' if stats['difference'] > 0 else ''
        print(f"  {cls_name}: 旧{stats['old']} → 新{stats['new']} ({diff_sign}{stats['difference']})")

def main():
    """主函数：解析命令行参数"""
    
    parser = argparse.ArgumentParser(description="基于YOLOv8的图像检测与对比系统")
    
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
                       help="预处理模板路径")
    parser.add_argument("--output", "-out", type=str,
                       default="./detection_results",
                       help="输出目录")
    parser.add_argument("--window", "-w", type=int,
                       default=512,
                       help="滑动窗口大小")
    parser.add_argument("--stride", "-s", type=int,
                       default=256,
                       help="滑动步长")
    parser.add_argument("--ignore-top", "-it", type=int,
                       default=700,
                       help="忽略顶部像素数")
    parser.add_argument("--no-refine", action="store_true",
                       help="不进行精细检测")
    parser.add_argument("--no-images", action="store_true",
                       help="不保存检测结果图像")
    parser.add_argument("--no-report", action="store_true",
                       help="不保存对比报告")
    
    args = parser.parse_args()
    
    # 运行处理流程
    process_images_pipeline(
        old_image_path=args.old,
        new_image_path=args.new,
        model_path=args.model,
        template_path=args.template,
        output_dir=args.output,
        window_size=args.window,
        stride=args.stride,
        ignore_top=args.ignore_top,
        refine_clusters=not args.no_refine,
        save_images=not args.no_images,
        save_report=not args.no_report
    )

if __name__ == "__main__":
    main()