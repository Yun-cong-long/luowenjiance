"""
图像处理流程脚本
功能：压缩图像 → 模板匹配 → 计算比例 → 分割原图 → 重新拼接
"""

from image_utils import compress_image, template_matching, image_split_recombine
import argparse

def process_image_pipeline(
    input_image_path, 
    template_path, 
    compression_ratio=6.25, 
    output_dir="./output",
    output_suffix="_recombined",
    match_threshold=0.8,
    draw_match=False
):
    """
    完整的图像处理流程
    
    参数:
        input_image_path: 输入图像路径
        template_path: 模板图像路径
        compression_ratio: 压缩比例（百分比）
        output_dir: 输出目录
        output_suffix: 输出文件后缀
        match_threshold: 模板匹配阈值（0-1）
        draw_match: 是否绘制匹配框
    """
    
    print("=" * 50)
    print(f"开始处理图像: {input_image_path}")
    print("=" * 50)
    
    # 1. 压缩图像（用于模板匹配）
    print(f"\n1. 压缩图像（压缩比例: {compression_ratio}%）...")
    try:
        compressed_img = compress_image(
            input_path=input_image_path,
            scale_percent=compression_ratio,
            output=False
        )
        print("   压缩完成！")
    except Exception as e:
        print(f"   压缩失败: {e}")
        return None, None
    
    # 2. 模板匹配
    print(f"\n2. 模板匹配（模板: {template_path}）...")
    try:
        left_x, left_ratio, right_ratio = template_matching(
            large_image=compressed_img,
            template_path=template_path,
            threshold=match_threshold,
            draw_match=draw_match,
            save_path=f"{output_dir}/match_result.jpg" if draw_match else None
        )
        
        if left_x is None or left_ratio is None:
            print(f"   模板匹配失败！匹配度低于阈值: {match_threshold}")
            return None, None
            
        print(f"   匹配成功！")
        print(f"   匹配位置: x={left_x}")
        print(f"   左侧比例: {left_ratio:.4f} ({left_ratio*100:.2f}%)")
        print(f"   右侧比例: {right_ratio:.4f} ({right_ratio*100:.2f}%)")
        
    except Exception as e:
        print(f"   模板匹配失败: {e}")
        return None, None
    
    # 3. 分割和重组原图（注意：使用原始图像，不是压缩后的）
    print(f"\n3. 分割和重组原图（分割比例: {left_ratio:.4f}）...")
    try:
        # 生成输出路径
        import os
        from pathlib import Path
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成输出文件名
        input_path = Path(input_image_path)
        output_filename = f"{input_path.stem}{output_suffix}{input_path.suffix}"
        output_path = Path(output_dir) / output_filename
        
        # 分割重组
        recombined_img, saved_path = image_split_recombine(
            input_path=input_image_path,
            split_ratio=left_ratio,
            save_output=True,
            output_path=str(output_path)
        )
        
        print(f"   分割重组完成！")
        print(f"   输出文件: {saved_path}")
        
        return recombined_img, saved_path
        
    except Exception as e:
        print(f"   分割重组失败: {e}")
        return None, None

def main():
    """主函数：解析命令行参数并执行处理流程"""
    
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="图像处理流程：压缩→匹配→分割→重组")
    
    # 添加命令行参数
    parser.add_argument(
        "--input", "-i", 
        type=str, 
        required=True,
        help="输入图像路径"
    )
    
    parser.add_argument(
        "--template", "-t", 
        type=str, 
        default="./moban/01_moban.bmp",
        help="模板图像路径（默认: ./moban/01_moban.bmp）"
    )
    
    parser.add_argument(
        "--ratio", "-r", 
        type=float, 
        default=6.25,
        help="压缩比例（百分比，默认: 6.25）"
    )
    
    parser.add_argument(
        "--output", "-o", 
        type=str, 
        default="./output",
        help="输出目录（默认: ./output）"
    )
    
    parser.add_argument(
        "--suffix", "-s", 
        type=str, 
        default="_recombined",
        help="输出文件后缀（默认: _recombined）"
    )
    
    parser.add_argument(
        "--threshold", "-th", 
        type=float, 
        default=0.8,
        help="模板匹配阈值（0-1，默认: 0.8）"
    )
    
    parser.add_argument(
        "--draw-match", "-d", 
        action="store_true",
        help="绘制匹配框并保存"
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 执行图像处理流程
    result_img, output_path = process_image_pipeline(
        input_image_path=args.input,
        template_path=args.template,
        compression_ratio=args.ratio,
        output_dir=args.output,
        output_suffix=args.suffix,
        match_threshold=args.threshold,
        draw_match=args.draw_match
    )
    
    if result_img is not None and output_path is not None:
        print(f"\n✓ 处理完成！结果已保存到: {output_path}")
    else:
        print(f"\n✗ 处理失败！")

if __name__ == "__main__":
    # 主程序入口
    main()

# 使用示例（可以作为独立函数调用）：
"""
# 示例1：使用命令行参数
# python yasuo_and_pipei.py --input ./luowen/L219/04.bmp --template ./moban/01_moban.bmp

# 示例2：使用函数调用
result, path = process_image_pipeline(
    input_image_path="./luowen/L219/04.bmp", # 输入图像地址
    template_path="./moban/01_moban.bmp",  # 模板地址
    compression_ratio=6.25,
    output_dir="./output",
    output_suffix="_processed",
    match_threshold=0.8
)
"""