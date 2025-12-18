import numpy as np
from PIL import Image
import random
import math
from scipy.cluster.vq import kmeans2, vq

class Config:
    def __init__(self, k_colors=None, k_seed=42):
        self.k_colors = k_colors
        self.k_seed = k_seed
        self.max_kmeans_iterations = 15
        self.peak_threshold_multiplier = 0.2
        self.peak_distance_filter = 4
        self.walker_search_window_ratio = 0.35
        self.walker_min_search_window = 2.0
        self.walker_strength_threshold = 0.5
        self.min_cuts_per_axis = 4
        self.fallback_target_segments = 64
        self.max_step_ratio = 3.0

def auto_detect_k(opaque_pixels, max_k=64):
    """自适应识别颜色数量：识别视觉主色并过滤杂色。不是很准"""
    if len(opaque_pixels) == 0:
        return 2
    
    sample = opaque_pixels[np.random.choice(len(opaque_pixels), min(len(opaque_pixels), 50000), replace=False)]
    
    # 统计频率
    packed = sample[:, 0].astype(np.int32) << 16 | sample[:, 1].astype(np.int32) << 8 | sample[:, 2].astype(np.int32)
    unique_vals, counts = np.unique(packed, return_counts=True)
    
    # 频率排序
    sort_idx = np.argsort(-counts)
    unique_rgbs = np.zeros((len(unique_vals), 3))
    unique_rgbs[:, 0], unique_rgbs[:, 1], unique_rgbs[:, 2] = (unique_vals >> 16) & 0xFF, (unique_vals >> 8) & 0xFF, unique_vals & 0xFF
    unique_rgbs = unique_rgbs[sort_idx]
    counts = counts[sort_idx]

    # 合并视觉相近的颜色，35-50范围调节
    dist_threshold_sq = 35 ** 2
    principal_colors = []
    coverage = 0
    target_coverage = len(sample) * 0.98 # 大部分颜色

    for i in range(len(unique_rgbs)):
        curr = unique_rgbs[i]
        if not principal_colors or np.min(np.sum((np.array(principal_colors) - curr)**2, axis=1)) > dist_threshold_sq:
            principal_colors.append(curr)
            coverage += counts[i]
        if len(principal_colors) >= max_k or coverage >= target_coverage:
            break
            
    return max(2, len(principal_colors))

def quantize_image_optimized(img_np, config):
    h, w = img_np.shape[:2]
    alpha = img_np[:, :, 3]
    opaque_mask = alpha > 0
    opaque_pixels = img_np[opaque_mask][:, :3].astype(np.float32)

    if len(opaque_pixels) == 0:
        return img_np, np.zeros((h, w), dtype=np.int32), np.zeros((1, 4))

    final_k = config.k_colors or auto_detect_k(opaque_pixels)
    
    k = min(final_k, len(opaque_pixels))
    print(f"k_colors is set to {k}({final_k})")
    centroids, _ = kmeans2(opaque_pixels, k, iter=config.max_kmeans_iterations, minit='points', seed=config.k_seed)
    
    all_rgb = img_np[:, :, :3].reshape(-1, 3).astype(np.float32)
    label_indices, _ = vq(all_rgb, centroids)
    index_map = label_indices.reshape(h, w).astype(np.int32)

    palette = np.column_stack((centroids, np.full(len(centroids), 255))).astype(np.uint8)
    quantized_rgb = np.dstack((palette[index_map][:,:,:3], alpha))
    return quantized_rgb, index_map, palette

def compute_profiles_vectorized(img_np):
    gray = 0.299 * img_np[:,:,0] + 0.587 * img_np[:,:,1] + 0.114 * img_np[:,:,2]
    gray[img_np[:,:,3] == 0] = 0.0
    col_proj = np.pad(np.sum(np.abs(gray[:, 2:] - gray[:, :-2]), axis=0), 1, mode='edge')
    row_proj = np.pad(np.sum(np.abs(gray[2:, :] - gray[:-2, :]), axis=1), 1, mode='edge')
    return col_proj.tolist(), row_proj.tolist()

def estimate_step_size(profile, config):
    p = np.array(profile)
    if np.max(p) == 0: return None
    peaks = np.where((p[1:-1] > np.max(p)*config.peak_threshold_multiplier) & (p[1:-1] >= p[:-2]) & (p[1:-1] >= p[2:]))[0] + 1
    if len(peaks) < 2: return None
    clean_peaks = [peaks[0]]
    for pk in peaks[1:]:
        if pk - clean_peaks[-1] >= config.peak_distance_filter: clean_peaks.append(pk)
    if len(clean_peaks) < 2: return None
    return float(np.median(np.diff(clean_peaks)))

def walk(profile, step_size, limit, config):
    cuts, curr = [0], 0.0
    win = max(step_size * config.walker_search_window_ratio, config.walker_min_search_window)
    mean_v = np.mean(profile)
    while curr < limit:
        target = curr + step_size
        if target >= limit:
            cuts.append(limit); break
        start, end = max(int(target - win), int(curr + 1)), min(int(target + win), limit)
        if end <= start:
            curr = target; cuts.append(int(target)); continue
        window = profile[start:end]
        max_idx = np.argmax(window)
        if window[max_idx] > mean_v * config.walker_strength_threshold:
            curr = float(start + max_idx); cuts.append(int(curr))
        else:
            curr = target; cuts.append(int(target))
    return sorted(list(set(cuts)))

def snap_uniform_cuts(profile, limit, target_step, config, min_req):
    cells = min(max(int(round(limit / target_step)), min_req - 1), limit)
    cell_w = limit / cells
    win = max(cell_w * config.walker_search_window_ratio, config.walker_min_search_window)
    mean_v = np.mean(profile) if profile else 0
    cuts = [0]
    for i in range(1, cells):
        target = cell_w * i
        prev = cuts[-1]
        start, end = max(prev + 1, int(target - win)), min(limit - 1, int(target + win))
        if end < start:
            cuts.append(min(prev + 1, limit - 1))
        else:
            window = profile[start:end+1]
            if np.max(window) > mean_v * config.walker_strength_threshold:
                cuts.append(start + np.argmax(window))
            else:
                cuts.append(min(max(int(round(target)), prev + 1), limit - 1))
    cuts.append(limit)
    return sorted(list(set(cuts)))

def stabilize_both_axes(px, py, raw_cx, raw_ry, w, h, config):
    # 移除强制 step_x == step_y
    def get_stab(prof, cuts, lim):
        c = sorted(list(set([0, lim] + [int(x) for x in cuts])))
        min_r = min(max(config.min_cuts_per_axis, 2), lim + 1)
        if len(c) >= min_r: return c
        t_step = lim / config.fallback_target_segments if config.fallback_target_segments > 1 else 10.0
        return snap_uniform_cuts(prof, lim, t_step, config, min_r)
    
    return get_stab(px, raw_cx, w), get_stab(py, raw_ry, h)

def resample_optimized(index_map, palette, alpha_map, cols, rows):
    out_w, out_h = len(cols) - 1, len(rows) - 1
    output_data = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    for y in range(out_h):
        ys, ye = rows[y], rows[y+1]
        for x in range(out_w):
            xs, xe = cols[x], cols[x+1] 
            idx_block, alpha_block = index_map[ys:ye, xs:xe], alpha_map[ys:ye, xs:xe]
            if idx_block.size == 0: continue
            if np.mean(alpha_block < 128) > 0.5:
                output_data[y, x] = [0, 0, 0, 0]
            else:
                opaque_indices = idx_block[alpha_block > 128]
                target = opaque_indices if opaque_indices.size > 0 else idx_block
                output_data[y, x] = palette[np.argmax(np.bincount(target.ravel()))]
    return Image.fromarray(output_data, "RGBA")

def process_image(input_bytes, k_colors=None):
    from io import BytesIO
    config = Config(k_colors=k_colors)
    img = Image.open(BytesIO(input_bytes)).convert("RGBA")
    img_np = np.array(img)
    w, h = img.size
    
    quantized_np, index_map, palette = quantize_image_optimized(img_np, config)
    px, py = compute_profiles_vectorized(quantized_np)
    
    sx = estimate_step_size(px, config) or (w / config.fallback_target_segments)
    sy = estimate_step_size(py, config) or (h / config.fallback_target_segments)

    raw_cx = walk(px, sx, w, config)
    raw_ry = walk(py, sy, h, config)
    
    cols, rows = stabilize_both_axes(px, py, raw_cx, raw_ry, w, h, config)
    result_img = resample_optimized(index_map, palette, img_np[:,:,3], cols, rows)
    
    out_io = BytesIO()
    result_img.save(out_io, format="PNG")
    return out_io.getvalue()