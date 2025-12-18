import numpy as np
from PIL import Image
from scipy.spatial import KDTree
from io import BytesIO

class PixelStylist:
    """
    针对已生成的像素图进行二次风格化加工。
    支持：调色板映射、抖动、CRT扫描线等。
    """
    
    # 经典调色板预设
    PALETTES = {
        'gameboy': [[15, 56, 15], [48, 98, 48], [139, 172, 15], [155, 188, 15]],
        'retro_pc': [[0,0,0], [255,255,255], [0,255,0], [255,0,0], [0,0,255], [255,255,0]],
        'cyberpunk': [[45, 15, 84], [171, 31, 101], [255, 79, 105], [51, 104, 220], [73, 231, 236]],
        'pico8': [[0,0,0], [29,43,83], [126,37,83], [0,135,81], [171,82,54], [95,87,79], [194,195,199], [255,241,232],
                  [255,0,77], [255,163,0], [255,236,39], [0,228,54], [41,173,255], [131,118,156], [255,119,168], [255,204,170]]
    }

    # Bayer 矩阵用于有序抖动 (4x4)
    BAYER_MATRIX = np.array([
        [ 0,  8,  2, 10],
        [12,  4, 14,  6],
        [ 3, 11,  1,  9],
        [15,  7, 13,  5]
    ]) / 16.0

    @staticmethod
    def apply_palette(img_np, palette):
        """将图像映射到指定的调色板"""
        h, w, c = img_np.shape
        tree = KDTree(palette)
        # 仅处理 RGB 部分
        pixels = img_np[:, :, :3].reshape(-1, 3)
        _, indices = tree.query(pixels)
        
        new_pixels = np.array(palette)[indices].reshape(h, w, 3)
        if c == 4:
            return np.dstack((new_pixels.astype(np.uint8), img_np[:, :, 3]))
        return new_pixels.astype(np.uint8)

    @classmethod
    def apply_dithering(cls, img_np, palette, spread=0.5):
        """抖动滤镜"""
        img = img_np.astype(np.float32)
        h, w, _ = img.shape
        
        bayer = np.tile(cls.BAYER_MATRIX, (h // 4 + 1, w // 4 + 1))[:h, :w]
        
        for i in range(3):
            img[:, :, i] += (bayer - 0.5) * spread * 255
            
        img = np.clip(img, 0, 255)
        return cls.apply_palette(img, palette)

    @staticmethod
    def apply_scanlines(img_np, intensity=0.3):
        """模拟 CRT 扫描线效果"""
        res = img_np.copy().astype(np.float32)
        # 每隔一行降低亮度
        res[1::2, :, :3] *= (1.0 - intensity)
        return np.clip(res, 0, 255).astype(np.uint8)

def apply_style(input_bytes, style_name='pico8', use_dither=True, use_scanlines=False):

    img = Image.open(BytesIO(input_bytes)).convert("RGBA")
    img_np = np.array(img)
    
    stylist = PixelStylist()
    palette = stylist.PALETTES.get(style_name, stylist.PALETTES['pico8'])
    
    # 1. 应用抖动或普通映射
    if use_dither:
        res = stylist.apply_dithering(img_np, palette)
    else:
        res = stylist.apply_palette(img_np, palette)
        
    # 2. 应用后处理滤镜
    if use_scanlines:
        res = stylist.apply_scanlines(res)
        
    out_io = BytesIO()
    Image.fromarray(res).save(out_io, format="PNG")
    return out_io.getvalue()