"""生成多样化的抽象形状图片 v3

相比 v2，v3 的核心改进：
- 不再每张图独立随机，而是用 4 维潜在空间在参数空间中走出一条平滑连续路径
- 相邻序号的图片结构相似、渐进变化，便于观察者归纳共性
- 远距离图片可能位于参数空间的不同区域，仍保持全局多样性

v1 是静态模板，v2 是独立随机，v3 是连续流形。
"""

import numpy as np
from dataclasses import dataclass
from PIL import Image, ImageDraw
from scipy.interpolate import make_interp_spline
import os
import random
import math

# ============================================================
# 全局配置
# ============================================================
NUM_IMAGES = 100
IMAGE_SIZE = 256
RANDOM_SEED = 44
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "images_v3")

# 潜在空间：4 个维度以不同频率在 [0, NUM_IMAGES) 上变化
NUM_LATENTS = 4
FREQUENCIES = [0.8, 2.3, 4.7, 8.1]   # cycles / 100 张图，非整数保证路径不重复
PHASES = [0.0, 0.5, 1.2, 2.8]

# ============================================================
# 调色板（所有图片背景为白色，仅从调色板取填充色）
# ============================================================
PALETTES = [
    [(219, 112, 147), (255, 160, 180), (180, 80, 120)],   # 粉色系
    [(100, 149, 237), (135, 180, 250), (65, 105, 200)],    # 蓝色系
    [(255, 165, 0), (255, 200, 80), (200, 130, 0)],        # 橙色系
    [(60, 179, 113), (100, 210, 150), (40, 140, 80)],      # 绿色系
    [(147, 112, 219), (170, 140, 240), (120, 80, 190)],    # 紫色系
    [(255, 99, 71), (255, 140, 100), (200, 70, 50)],       # 红色系
    [(255, 215, 0), (255, 235, 80), (200, 180, 0)],        # 黄色系
    [(0, 150, 136), (50, 180, 160), (0, 120, 100)],        # 青绿色系
    [(244, 164, 96), (255, 200, 150), (210, 130, 70)],     # 沙色系
    [(135, 206, 235), (180, 220, 250), (80, 170, 210)],    # 天蓝色系
]


# ============================================================
# 数据类
# ============================================================
@dataclass
class ImageConfig:
    num_axes: int
    axes_angles: list
    symmetry_mode: str
    distance_func: str
    num_layers: int
    colors: list
    rotation: float
    scale: float
    rotational_folds: int = 0


# ============================================================
# 潜在空间 → 配置映射
# ============================================================
def compute_latents(idx: int) -> list:
    """计算 4 维潜在向量，随 idx 平滑变化，值域 [-1, 1]。"""
    latents = []
    for d in range(NUM_LATENTS):
        val = np.sin(2 * np.pi * FREQUENCIES[d] * idx / NUM_IMAGES + np.pi * PHASES[d])
        latents.append(float(val))
    return latents


def generate_angles(num_axes: int, strategy: str, seed: int) -> list:
    """生成 n 个角度（0°~180°），用 seed 保证确定性和连续性。"""
    if strategy == "uniform":
        return np.linspace(0, 180, num_axes).tolist()

    local_rng = random.Random(seed)
    if strategy == "random":
        return sorted(local_rng.uniform(0, 180) for _ in range(num_axes))

    # clustered
    center = 30 + local_rng.uniform(0, 120)
    spread = 12 + local_rng.uniform(0, 38)
    n = max(num_axes, 3)
    inner = [max(0, min(180, local_rng.gauss(center, spread))) for _ in range(n - 2)]
    return sorted([0.0, 180.0] + inner)


def generate_config_from_latents(latents: list, idx: int) -> ImageConfig:
    """将 4 维潜在向量映射为完整 ImageConfig。"""
    l0, l1, l2, l3 = latents

    # ----- 结构复杂度：轴数 + 层数（慢速变化） -----
    num_axes = max(4, min(14, round(4 + (l0 + 1) * 5)))

    if l0 < -0.3:
        num_layers = 1
    elif l0 < 0.3:
        num_layers = 2
    else:
        num_layers = 3

    # ----- 形状家族：对称模式 + 距离函数（中速变化） -----
    if l1 < -0.3:
        symmetry_mode = "mirror"
    elif l1 < 0.3:
        symmetry_mode = "asymmetric"
    else:
        symmetry_mode = "rotational"

    dist_mix = l1 * 0.7 + l2 * 0.3
    if dist_mix < -0.3:
        distance_func = "cos"
    elif dist_mix < 0.3:
        distance_func = "noise"
    else:
        distance_func = "polynomial"

    # ----- 几何细节：角度策略 + 旋转（中快速变化） -----
    if l2 < -0.3:
        angle_strategy = "uniform"
    elif l2 < 0.3:
        angle_strategy = "random"
    else:
        angle_strategy = "clustered"

    rotation = (l2 + 1) / 2 * 360

    # ----- 外观：调色板 + 缩放（快速变化） -----
    palette = PALETTES[int((l3 + 1) / 2 * len(PALETTES)) % len(PALETTES)]
    scale = 0.75 + (l3 + 1) / 2 * 0.25

    rotational_folds = max(2, min(6, round(2 + (l1 + 1) * 2))) if symmetry_mode == "rotational" else 0

    # ----- 角度序列（seed = idx 保证连续） -----
    axes_angles = generate_angles(num_axes, angle_strategy, idx * 7 + 13)

    colors = [palette[i % len(palette)] for i in range(num_layers)]

    return ImageConfig(
        num_axes=num_axes,
        axes_angles=axes_angles,
        symmetry_mode=symmetry_mode,
        distance_func=distance_func,
        num_layers=num_layers,
        colors=colors,
        rotation=rotation,
        scale=scale,
        rotational_folds=rotational_folds,
    )


# ============================================================
# 骨架计算（距离函数池，seed 驱动保证连续性）
# ============================================================
def compute_skeleton(num_axes: int, distance_func: str, seed: float) -> list:
    """根据距离函数类型计算骨架。seed 连续则输出连续。"""
    local_rng = random.Random(int(seed * 1000) & 0x7FFFFFFF)
    distances = []

    if distance_func == "cos":
        omega = 2 * np.pi * local_rng.uniform(0.001, 0.012)
        phases = [local_rng.uniform(0, 2 * np.pi) for _ in range(num_axes)]
        for k in range(num_axes):
            v = abs(np.cos(seed * omega * (np.log(k + 1) + 1) + phases[k]))
            distances.append(0.1 + 0.8 * v)

    elif distance_func == "noise":
        omega = 2 * np.pi * local_rng.uniform(0.002, 0.015)
        num_h = local_rng.randint(2, 5)
        phases = [[local_rng.uniform(0, 2 * np.pi) for _ in range(num_axes)] for _ in range(num_h)]
        freqs = [local_rng.uniform(0.5, 2.0) for _ in range(num_h)]
        for k in range(num_axes):
            v = sum(
                np.sin(seed * omega * freqs[h] * (np.log(k + 1) + 1) + phases[h][k])
                for h in range(num_h)
            )
            v = abs(v / num_h)
            distances.append(0.1 + 0.8 * v)

    elif distance_func == "polynomial":
        n = max(num_axes, 2)
        t = np.linspace(0, 1, n)
        a = local_rng.uniform(-1.5, 1.5)
        b = local_rng.uniform(-1.5, 1.5)
        c = local_rng.uniform(0.3, 1.0)
        phase_shift = seed * 0.08
        for i in range(n):
            v = abs(a * t[i] ** 2 + b * t[i] + c + 0.3 * np.sin(phase_shift + t[i] * np.pi * 2))
            v = min(1.0, v)
            distances.append(0.1 + 0.8 * v)

    return distances


# ============================================================
# 坐标转换 & 平滑 & 旋转
# ============================================================
def skeleton_to_points(skeleton: list, center: tuple, max_radius: float, angles: list) -> list:
    cx, cy = center
    points = []
    for i, angle in enumerate(angles):
        angle_rad = np.deg2rad(angle)
        r = skeleton[i] * max_radius
        x = cx + r * np.sin(angle_rad)
        y = cy - r * np.cos(angle_rad)
        points.append((float(x), float(y)))
    return points


def draw_smooth_curve(points: list) -> list:
    n = len(points)
    if n < 2:
        return points
    if n == 2:
        x0, y0 = points[0]
        x1, y1 = points[1]
        return [(x0 + t / 99 * (x1 - x0), y0 + t / 99 * (y1 - y0)) for t in range(100)]

    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    t = np.linspace(0, 1, n)
    t_new = np.linspace(0, 1, 200)
    spline = make_interp_spline(t, np.column_stack([x, y]), k=3, bc_type="natural")
    smooth = spline(t_new)
    return [(float(p[0]), float(p[1])) for p in smooth]


def rotate_points(points: list, center: tuple, angle_deg: float) -> list:
    cx, cy = center
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    return [
        (cx + (x - cx) * cos_a - (y - cy) * sin_a,
         cy + (x - cx) * sin_a + (y - cy) * cos_a)
        for x, y in points
    ]


# ============================================================
# 核心绘图
# ============================================================
def draw_abstract_shape(size: int, config: ImageConfig, idx: int) -> Image.Image:
    img = Image.new("RGB", (size, size), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    cx = cy = size / 2

    for layer in range(config.num_layers):
        layer_scale = config.scale * (1.0 - 0.08 * layer)
        layer_color = config.colors[layer]
        max_radius = size / 2 * 0.85 * layer_scale

        polygon = _build_layer_polygon(config, idx, layer, cx, cy, max_radius)

        if config.rotation != 0:
            polygon = rotate_points(polygon, (cx, cy), config.rotation)

        draw.polygon(polygon, fill=layer_color)

    return img


def _build_layer_polygon(config: ImageConfig, idx: int, layer: int, cx: float, cy: float, max_radius: float) -> list:
    """构建单层多边形。seed = idx * 100 + layer 保证相邻图片的骨架连续变化。"""
    layer_seed = idx * 100 + layer

    if config.symmetry_mode == "mirror":
        skeleton = compute_skeleton(config.num_axes, config.distance_func, layer_seed)
        right_pts = skeleton_to_points(skeleton, (cx, cy), max_radius, config.axes_angles)
        right_smooth = draw_smooth_curve(right_pts)
        left_smooth = [(2 * cx - x, y) for x, y in reversed(right_smooth)]
        return right_smooth + left_smooth

    if config.symmetry_mode == "asymmetric":
        skeleton_r = compute_skeleton(config.num_axes, config.distance_func, layer_seed)
        skeleton_l = compute_skeleton(config.num_axes, config.distance_func, layer_seed + 50000)
        right_pts = skeleton_to_points(skeleton_r, (cx, cy), max_radius, config.axes_angles)
        left_pts = skeleton_to_points(skeleton_l, (cx, cy), max_radius, config.axes_angles)
        left_pts = [(2 * cx - x, y) for x, y in left_pts]
        right_smooth = draw_smooth_curve(right_pts)
        left_smooth = draw_smooth_curve(list(reversed(left_pts)))
        return right_smooth + left_smooth

    if config.symmetry_mode == "rotational":
        folds = max(config.rotational_folds, 2)
        half_wedge = max(config.num_axes // 2, 3)
        wedge_angles = np.linspace(0, 180.0 / folds, half_wedge).tolist()
        skeleton = compute_skeleton(half_wedge, config.distance_func, layer_seed)
        right_pts = skeleton_to_points(skeleton, (cx, cy), max_radius, wedge_angles)
        right_smooth = draw_smooth_curve(right_pts)
        left_smooth = [(2 * cx - x, y) for x, y in reversed(right_smooth)]
        wedge = right_smooth + left_smooth
        all_points = []
        for fold in range(folds):
            all_points.extend(rotate_points(wedge, (cx, cy), fold * 360.0 / folds))
        return all_points

    return []


# ============================================================
# 主流程
# ============================================================
def generate_and_save_images():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for idx in range(NUM_IMAGES):
        latents = compute_latents(idx)
        config = generate_config_from_latents(latents, idx)
        img = draw_abstract_shape(IMAGE_SIZE, config, idx)
        path = os.path.join(OUTPUT_DIR, f"abstract_{idx:03d}.png")
        img.save(path)

        if (idx + 1) % 25 == 0:
            print(f"已生成 {idx + 1}/{NUM_IMAGES} 张图片")

    print(f"全部完成！图片保存在 {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_and_save_images()
