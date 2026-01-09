# 图像处理工具包  
**image_utils.py**中包含三个核心功能：

1. 按比例压缩图像: compress_image

2. 模板匹配并计算比例: template_matching

3. 图像分割重组: image_split_recombine

---

**predesl.py**中*process_image_pipeline*调用了*image_utils.py*中的函数，整合函数功能。

调用方式：
```python
# 使用所有参数
result_img, output_path = process_image_pipeline(
    input_image_path="./luowen/L219/04.bmp",  # 必需
    template_path="./moban/01_moban.bmp",      # 指定模板
    compression_ratio=12.5,                     # 压缩到12.5%
    output_dir="./processed_results",           # 自定义输出目录
    output_suffix="_final",                     # 自定义后缀
    match_threshold=0.85,                       # 提高匹配阈值
    draw_match=True                             # 保存匹配框图像
)
```