import numpy as np
from PIL import Image
from io import BytesIO

class PixelUpscaler:

    @staticmethod
    def upscale(input_bytes, target_width=None, target_height=None, scale_factor=None):
        """
        核心放大功能
        :param input_bytes: 原始小图的字节数据
        :param target_width: 目标宽度（例如 1920）
        :param target_height: 目标高度（例如 1080）
        :param scale_factor: 放大倍数（例如 10，高优先级）
        """
        img = Image.open(BytesIO(input_bytes)).convert("RGBA")
        orig_w, orig_h = img.size

        # 计算目标尺寸
        if scale_factor:
            final_w = orig_w * scale_factor
            final_h = orig_h * scale_factor
        elif target_width and target_height:
            final_w = target_width
            final_h = target_height
        else:
            # 默认放大 10 倍
            final_w, final_h = orig_w * 10, orig_h * 10

        upscaled_img = img.resize((final_w, final_h), resample=Image.NEAREST)
        
        return upscaled_img

    @staticmethod
    def upscale_with_grid(input_bytes, scale_factor=10, grid_opacity=0.1):
        """
        带网格缝隙（老式显示器）
        :param grid_opacity: 缝隙的透明度 (0.0 ~ 1.0)
        """
        img = Image.open(BytesIO(input_bytes)).convert("RGBA")
        orig_w, orig_h = img.size
        
        upscaled = img.resize((orig_w * scale_factor, orig_h * scale_factor), resample=Image.NEAREST)
        data = np.array(upscaled).astype(np.float32)
        
        for i in range(0, orig_w * scale_factor, scale_factor):
            data[:, i:i+1, :3] *= (1.0 - grid_opacity) # 垂直线
            
        for j in range(0, orig_h * scale_factor, scale_factor):
            data[j:j+1, :, :3] *= (1.0 - grid_opacity) # 水平线
            
        return Image.fromarray(np.clip(data, 0, 255).astype(np.uint8))


if __name__ == "__main__":
    from tool import save_pil_image, open_image
    
    input_file = "small_pixel_art.png"
    output_file = "upscaled_hd_art.png"
    
    try:
        small_data = open_image(input_file)
        upscaler = PixelUpscaler()
        
        # 直接硬放大到指定尺寸 
        # hd_img = upscaler.upscale(small_data, target_width=1920, target_height=1080)
        
        # 按倍数放大 (比如放大 10 倍)，并带有一点像素缝隙感
        hd_img = upscaler.upscale_with_grid(small_data, scale_factor=10, grid_opacity=0.15)
        
        save_pil_image(hd_img, output_file)
        
    except FileNotFoundError:
        print("no files")