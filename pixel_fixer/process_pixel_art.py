import os
import numpy as np
import scipy.signal
import scipy.ndimage
from collections import Counter
from PIL import Image, ImageSequence


def auto_detect_block_size(frames, min_size=2, max_size=50):
    """ 通过分析图像梯度和形态学处理，自动检测块大小。 """
    
    all_distances = []
    for i, frame in enumerate(frames):
        # --- 预处理和梯度计算 ---
        gray_img = frame.convert('L')
        data = np.array(gray_img, dtype=np.float32)
        data = scipy.ndimage.gaussian_filter(data, sigma=1.2)

        sobel_x = scipy.ndimage.sobel(data, axis=0)
        sobel_y = scipy.ndimage.sobel(data, axis=1)
        gradient = np.hypot(sobel_x, sobel_y)

        # --- 创建清晰的二值化边缘图 ---
        # 自适应阈值化：使用百分位数来确定强边缘
        if gradient.max() == 0: continue # 跳过纯色帧
        threshold = np.percentile(gradient, 85) # 85th percentile
        
        # 创建二值化边缘图
        edge_map = gradient > threshold
        
        # 形态学开运算：移除噪声
        structure = np.ones((2, 2))
        cleaned_edge_map = scipy.ndimage.binary_opening(edge_map, structure=structure, iterations=1)

        # --- 在净化后的边缘图上进行分析 ---
        height, width = cleaned_edge_map.shape
        
        # 水平分析
        h_projection = np.sum(cleaned_edge_map, axis=1)
        h_peaks, _ = scipy.signal.find_peaks(h_projection, height=1, distance=min_size - 1)
        if len(h_peaks) > 1: all_distances.extend(np.diff(h_peaks))

        # 垂直分析
        v_projection = np.sum(cleaned_edge_map, axis=0)
        v_peaks, _ = scipy.signal.find_peaks(v_projection, height=1, distance=min_size - 1)
        if len(v_peaks) > 1: all_distances.extend(np.diff(v_peaks))

    if not all_distances:
        print("警告：未能检测到足够的边缘来估算块大小。")
        return None

    filtered_distances = [d for d in all_distances if min_size <= d <= max_size]
    
    if not filtered_distances:
        print(f"警告：所有检测到的距离都在过滤范围之外 ({min_size}-{max_size})。")
        return None

    # 聚类中心
    rounded_distances = np.round(filtered_distances).astype(int)
    count = Counter(rounded_distances)
    if not count:
        return None

    estimated_size = count.most_common(1)[0][0]
    return int(estimated_size)


def color_distance_sq(c1, c2):
    """ 计算两个RGB颜色之间欧氏距离的平方。 """
    r1, g1, b1 = map(float, c1)
    r2, g2, b2 = map(float, c2)
    
    return (r1 - r2)**2 + (g1 - g2)**2 + (b1 - b2)**2


def find_clustered_color(block, similarity_threshold=60):
    """
    通过色彩聚类找到块的代表色，能有效合并杂色并平滑过渡。

    参数:
    - block (np.ndarray): 输入的像素块。
    - similarity_threshold (int): 颜色相似度阈值。距离小于此值的颜色会被聚类。

    返回:
    - tuple: (R, G, B) 格式的最终颜色。
    """
    if block.size == 0: return (0, 0, 0)

    pixels = block.reshape(-1, block.shape[-1])
    counts = Counter(map(tuple, pixels))
    if len(counts) == 1:
        return list(counts.keys())[0][:3]
    
    sorted_colors = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    
    clusters = []
    threshold_sq = similarity_threshold**2

    for color, count in sorted_colors:
        color_rgb = color[:3]
        merged = False
        for cluster in clusters:
            # 检查是否可以合并到现有簇
            if color_distance_sq(color_rgb, cluster['mean_color']) < threshold_sq:
                # 更新簇的加权平均色和总数
                total_count = cluster['count'] + count
                cluster['mean_color'] = tuple(
                    int((cluster['mean_color'][i] * cluster['count'] + color_rgb[i] * count) / total_count)
                    for i in range(3)
                )
                cluster['count'] = total_count
                merged = True
                break
        
        if not merged:
            clusters.append({'mean_color': color_rgb, 'count': count})
            
    if not clusters: return (0, 0, 0) # 极端情况
    dominant_cluster = max(clusters, key=lambda c: c['count'])
    
    # 返回最大簇的代表色
    return dominant_cluster['mean_color']

def process_single_frame_cluster(frame_data, block_size, similarity_threshold):
    """ 单帧处理函数，使用色彩聚类。 """
    height, width, _ = frame_data.shape
    output_full = np.zeros_like(frame_data)

    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            y_end = min(y + block_size, height)
            x_end = min(x + block_size, width)
            block = frame_data[y:y_end, x:x_end]
            
            final_color = find_clustered_color(block, similarity_threshold)
            
            output_full[y:y_end, x:x_end] = final_color
    
    # 生成缩放图
    scaled_height = (height + block_size - 1) // block_size
    scaled_width = (width + block_size - 1) // block_size
    scaled_img = Image.fromarray(output_full).resize((scaled_width, scaled_height), Image.Resampling.NEAREST)
    output_scaled = np.array(scaled_img)

    return output_full, output_scaled

def process_pixel_art(
    image_path: str, 
    block_size: int = None, 
    output_path: str = 'result.png',
    num_colors: int = None,
    color_similarity_threshold: int = 60
):

    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"错误：找不到文件 '{image_path}'")
        return
        
    is_animated = hasattr(img, 'is_animated') and img.is_animated
    original_frames_pil = [frame.copy().convert('RGB') for frame in ImageSequence.Iterator(img)]

    if block_size is None or block_size <= 0:
        block_size = auto_detect_block_size(original_frames_pil)
        if block_size is None: 
            print("自动检测失败，程序终止。请手动指定 `block_size`。")
            return
            
    print(f"图像尺寸: {img.width}×{img.height}, 帧数: {len(original_frames_pil)}")
    print(f"使用的像素块大小: {block_size}x{block_size}")

    base, ext = os.path.splitext(output_path)
    output_path_full = f"{base}_{block_size}_full{ext}"
    output_path_scaled = f"{base}_{block_size}_scaled{ext}"
    
    # --- 处理帧 ---
    final_frames_full = []
    for i, frame_pil in enumerate(original_frames_pil):
        frame_data = np.array(frame_pil)
        
        processed_full_data, _ = process_single_frame_cluster(
            frame_data, 
            block_size, 
            color_similarity_threshold
        )
        final_frames_full.append(Image.fromarray(processed_full_data))

    if num_colors:
        quantized_full = []
        for frame in final_frames_full:
            quantized_frame = frame.quantize(colors=num_colors, method=Image.Quantize.MEDIANCUT).convert('RGB')
            quantized_full.append(quantized_frame)
        final_frames_full = quantized_full

    # --- 从处理完的全尺寸图中生成缩放图 ---
    final_frames_scaled = []
    scaled_height = (img.height + block_size - 1) // block_size
    scaled_width = (img.width + block_size - 1) // block_size
    for frame in final_frames_full:
        # 使用NEAREST采样
        scaled_img = frame.resize((scaled_width, scaled_height), Image.Resampling.NEAREST)
        final_frames_scaled.append(scaled_img)


    # --- 保存 ---
    if is_animated:
        durations = [frame.info.get('duration', 100) for frame in ImageSequence.Iterator(img)]
        loop = img.info.get('loop', 0)
        final_frames_full[0].save(
            output_path_full, save_all=True, append_images=final_frames_full[1:],
            duration=durations, loop=loop, optimize=False
        )
        final_frames_scaled[0].save(
            output_path_scaled, save_all=True, append_images=final_frames_scaled[1:],
            duration=durations, loop=loop, optimize=False
        )
    else:
        final_frames_full[0].save(output_path_full)
        final_frames_scaled[0].save(output_path_scaled)
    print(f"结果已保存至: '{output_path_full}'")

if __name__ == '__main__':
    num_colors = None
    color_similarity_threshold=60
    input_file = 'example.jpeg'
    output_file = input_file.split('.')[0] + f'_{num_colors}.' + ('png' if input_file.split('.')[1] != 'gif' else 'gif')

    process_pixel_art(
        image_path=input_file,
        block_size=None, # None: 自动检测
        output_path=output_file,
        num_colors=num_colors,
        color_similarity_threshold=color_similarity_threshold
    )
