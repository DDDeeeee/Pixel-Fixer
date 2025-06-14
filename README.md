# Pixel-Fixer (像素修复)

<p align="center">
  <strong>Transform AI-generated "pseudo-pixel art" into authentic pixel art.</strong>
  <br>
  <em>将AI生成的“伪像素图”转换为纯正的像素艺术。</em>
</p>

![Showcase](https://github.com/DDDeeeee/Pixel-Fixer/blob/main/comparative_example.png)
*<p align="center">处理前后对比</p>*

## 😫 The Problem

AI生成的“伪像素图”存在以下问题：

- **不均匀的像素块**: “像素”的尺寸和形状不一；
- **模糊与抗锯齿**: 边缘存在不希望的模糊和斜边；
- **过多的颜色**: 一个“色块”内可能包含几十种极其相似的颜色。

## ✨ Features

- **网格检测**: 自动检测伪像素图的“像素块大小”；
- **色彩聚类**: 使用聚类来合并噪点并创建平滑、自然的颜色过渡；
- **色彩量化**: 可将最终的调色板精简到指定数量；
- **双格式输出**: 生成全尺寸图和“点对点”缩放图；
- **支持动图 (GIF)**: 逐帧处理GIF动图，并保证整个动画使用统一的网格尺寸和调色板。

## 🚀 Usage

```bash
pip install Pillow numpy scipy
```

```python
from pixel_fixer.process_pixel_art import *

num_colors = None
color_similarity_threshold=20
input_file = 'xxx.png'
output_file = input_file.split('.')[0] + f'_{num_colors}.' + ('png' if input_file.split('.')[1] != 'gif' else 'gif')

process_pixel_art(
    image_path='img_input/' + input_file,
    block_size=None, # None: 自动检测
    output_path='img_output/' + output_file,
    num_colors=num_colors,
    color_similarity_threshold=color_similarity_threshold
)
```
