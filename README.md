# Pixel-Fixer (像素修复)

<p align="center">
  <strong>Transform AI-generated "pseudo-pixel art" into authentic pixel art.</strong>
  <br>
  <em>将AI生成的“伪像素图”转换为真正的像素图。</em>
</p>

![Showcase](https://github.com/DDDeeeee/Pixel-Fixer/blob/main/img_input/img1.jpeg)
*<p align="center">原始图像>yp>*
![Showcase](https://github.com/DDDeeeee/Pixel-Fixer/blob/main/img_output/img1_upscaled.png)
*<p align="center">处理后图像</p>*
![Showcase](https://github.com/DDDeeeee/Pixel-Fixer/blob/main/img_output/img1_stylized_upscaled.png)
*<p align="center">风格化图像</p>*

## Updata

2. Dec 12, 2025
Referencing the following project, the relevant code has been upgraded: [pixel-snapper](https://github.com/Hugo-Dz/spritefusion-pixel-snapper), [pixelit](https://github.com/giventofly/pixelit).
1. Jun 14, 2025  
Upload project.

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
- **风格化滤镜**: 根据调色板进行风格化映射。

## 🚀 Usage
1. 配置环境
```bash
pip install Pillow numpy scipy
```
2. 执行代码
```python
from pathlib import Path
from tool import open_image, save_image, save_pil_image
from pixel_stylist import apply_style
from process_pixel_art import process_image
from pixel_upscaler import PixelUpscaler

upscaler = PixelUpscaler()

org_img_path = "img_input/img1.jpeg"
output_img_path = Path("img_output/img1.jpeg")

# 像素化
data = open_image(org_img_path)
result = process_image(data)
save_image(result, output_img_path.with_stem(output_img_path.stem + "_pix"))

# 高清化
hd_img = upscaler.upscale_with_grid(result, scale_factor=10, grid_opacity=0)
save_pil_image(hd_img, output_img_path.with_stem(output_img_path.stem + "_upscaled").with_suffix(".png"))

# 风格化
final_art = apply_style(result, style_name='pico8', use_dither=True, use_scanlines=True)
stylized_output_img_path = output_img_path.with_stem(output_img_path.stem + "_stylized")
save_image(final_art, stylized_output_img_path)

# 高清化
hd_stylized_img = upscaler.upscale_with_grid(final_art, scale_factor=10, grid_opacity=0)
upscaled_stylized_output_img_path = stylized_output_img_path.with_stem(stylized_output_img_path.stem + "_upscaled").with_suffix(".png")
save_pil_image(hd_stylized_img, upscaled_stylized_output_img_path)
```
3. 手动修复边缘、抖色、抗锯齿、错误色点等细节。
