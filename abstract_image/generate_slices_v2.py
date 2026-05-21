"""生成多样化的抽象形状图片 v2

相比 v1（generate_slices.py），v2 的核心改进：
- 每张图的结构参数独立随机（轴数、角度分布、对称模式、距离函数）
- 三种对称模式：镜像 / 非对称 / 旋转对称
- 三种距离函数：cos波 / 类噪声 / 多项式
- 多层形状叠加 + 多组调色板
"""

import numpy as np
from dataclasses import dataclass, field
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
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "images_v2")

# ============================================================
# 调色板
# ============================================================
PALETTES = [
    {
        "fills": [(219, 112, 147), (255, 160, 180), (180, 80, 120)],
        "bg": (255, 255, 255),
    },
    {
        "fills": [(100, 149, 237), (135, 180, 250), (65, 105, 200)],
        "bg": (240, 245, 255),
    },
    {
        "fills": [(255, 165, 0), (255, 200, 80), (200, 130, 0)],
        "bg": (255, 252, 245),
    },
    {
        "fills": [(60, 179, 113), (100, 210, 150), (40, 140, 80)],
        "bg": (245, 255, 248),
    },
    {
        "fills": [(147, 112, 219), (170, 140, 240), (120, 80, 190)],
        "bg": (250, 245, 255),
    },
    {
        "fills": [(255, 99, 71), (255, 140, 100), (200, 70, 50)],
        "bg": (255, 248, 245),
    },
    {
        "fills": [(255, 215, 0), (255, 235, 80), (200, 180, 0)],
        "bg": (255, 255, 248),
    },
    {
        "fills": [(0, 150, 136), (50, 180, 160), (0, 120, 100)],
        "bg": (248, 252, 250),
    },
    {
        "fills": [(244, 164, 96), (255, 200, 150), (210, 130, 70)],
        "bg": (25, 25, 35),
    },
    {
        "fills": [(135, 206, 235), (180, 220, 250), (80, 170, 210)],
        "bg": (20, 25, 35),
    },
]


# ============================================================
# 数据类
# ============================================================
@dataclass
class ImageConfig:
    num_axes: int
    axes_angles: list
    symmetry_mode: str      # "mirror" | "asymmetric" | "rotational"
    distance_func: str       # "cos" | "noise" | "polynomial"
    num_layers: int
    colors: list
    bg_color: tuple
    rotation: float
    scale: float
    rotational_folds: int = 0


# ============================================================
# 角度生成
# ============================================================
def generate_angles(n: int, strategy: str, rng: random.Random) -> list:
    """生成 n 个角度（0°~180°），支持三种分布策略。"""
    if strategy == "uniform":
        return np.linspace(0, 180, n).tolist()

    if strategy == "random":
        return sorted(rng.uniform(0, 180) for _ in range(n))

    # clustered: 角度集中在随机中心附近 + 首尾各保留一个端点
    center = rng.uniform(30, 150)
    spread = rng.uniform(12, 50)
    inner = [max(0, min(180, rng.gauss(center, spread))) for _ in range(n - 2)]
    return sorted([0.0, 180.0] + inner)


# ============================================================
# 随机配置生成
# ============================================================
def generate_random_config(rng: random.Random) -> ImageConfig:
    """为单张图片生成随机配置。"""
    symmetry_mode = rng.choice(["mirror", "asymmetric", "rotational"])
    num_axes = rng.randint(4, 14)
    distance_func = rng.choice(["cos", "noise", "polynomial"])
    num_layers = rng.choices([1, 2, 3], weights=[0.3, 0.4, 0.3])[0]
    palette = rng.choice(PALETTES)

    angle_strategy = rng.choice(["uniform", "random", "clustered"])
    axes_angles = generate_angles(num_axes, angle_strategy, rng)

    colors = [palette["fills"][i % len(palette["fills"])] for i in range(num_layers)]

    rotation = rng.uniform(0, 360)
    scale = rng.uniform(0.75, 1.0)
    rotational_folds = rng.randint(2, 6) if symmetry_mode == "rotational" else 0

    return ImageConfig(
        num_axes=num_axes,
        axes_angles=axes_angles,
        symmetry_mode=symmetry_mode,
        distance_func=distance_func,
        num_layers=num_layers,
        colors=colors,
        bg_color=(255, 255, 255),
        rotation=rotation,
        scale=scale,
        rotational_folds=rotational_folds,
    )


# ============================================================
# 骨架计算（距离函数池）
# ============================================================
def compute_skeleton(num_axes: int, distance_func: str, seed: float, rng: random.Random) -> list:
    """根据距离函数类型计算骨架（每个轴的径向距离）。"""
    distances = []

    if distance_func == "cos":
        omega = 2 * np.pi * rng.uniform(0.001, 0.012)
        phases = [rng.uniform(0, 2 * np.pi) for _ in range(num_axes)]
        for k in range(num_axes):
            v = abs(np.cos(seed * omega * (np.log(k + 1) + 1) + phases[k]))
            distances.append(0.1 + 0.8 * v)

    elif distance_func == "noise":
        omega = 2 * np.pi * rng.uniform(0.002, 0.015)
        num_h = rng.randint(2, 5)
        phases = [[rng.uniform(0, 2 * np.pi) for _ in range(num_axes)] for _ in range(num_h)]
        freqs = [rng.uniform(0.5, 2.0) for _ in range(num_h)]
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
        a = rng.uniform(-1.5, 1.5)
        b = rng.uniform(-1.5, 1.5)
        c = rng.uniform(0.3, 1.0)
        phase = seed * 0.08
        for i in range(n):
            v = abs(a * t[i] ** 2 + b * t[i] + c + 0.3 * np.sin(phase + t[i] * np.pi * 2))
            v = min(1.0, v)
            distances.append(0.1 + 0.8 * v)

    return distances


# ============================================================
# 坐标转换 & 平滑 & 旋转
# ============================================================
def skeleton_to_points(
    skeleton: list, center: tuple, max_radius: float, angles: list
) -> list:
    """将骨架距离列表转换为 (x, y) 坐标点。"""
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
    """三次样条插值生成平滑曲线。"""
    n = len(points)
    if n < 2:
        return points
    if n == 2:
        x0, y0 = points[0]
        x1, y1 = points[1]
        return [
            (x0 + t / 99 * (x1 - x0), y0 + t / 99 * (y1 - y0))
            for t in range(100)
        ]

    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])
    t = np.linspace(0, 1, n)
    t_new = np.linspace(0, 1, 200)
    spline = make_interp_spline(t, np.column_stack([x, y]), k=3, bc_type="natural")
    smooth = spline(t_new)
    return [(float(p[0]), float(p[1])) for p in smooth]


def rotate_points(points: list, center: tuple, angle_deg: float) -> list:
    """将点集绕 center 旋转 angle_deg 度。"""
    cx, cy = center
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    rotated = []
    for x, y in points:
        dx = x - cx
        dy = y - cy
        rotated.append((cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a))
    return rotated


# ============================================================
# 核心绘图
# ============================================================
def draw_abstract_shape(size: int, config: ImageConfig, rng: random.Random) -> Image.Image:
    """根据 ImageConfig 绘制单张抽象形状图片。"""
    img = Image.new("RGB", (size, size), config.bg_color)
    draw = ImageDraw.Draw(img)
    cx = cy = size / 2

    for layer in range(config.num_layers):
        layer_scale = config.scale * (1.0 - 0.08 * layer)
        layer_color = config.colors[layer]
        max_radius = size / 2 * 0.85 * layer_scale

        polygon = _build_layer_polygon(config, rng, cx, cy, max_radius)

        if config.rotation != 0:
            polygon = rotate_points(polygon, (cx, cy), config.rotation)

        draw.polygon(polygon, fill=layer_color)

    return img


def _build_layer_polygon(config: ImageConfig, rng: random.Random, cx: float, cy: float, max_radius: float) -> list:
    """构建单层多边形点序列。"""
    layer_seed = rng.random() * 100000

    if config.symmetry_mode == "mirror":
        skeleton = compute_skeleton(config.num_axes, config.distance_func, layer_seed, rng)
        right_pts = skeleton_to_points(skeleton, (cx, cy), max_radius, config.axes_angles)
        right_smooth = draw_smooth_curve(right_pts)
        left_smooth = [(2 * cx - x, y) for x, y in reversed(right_smooth)]
        return right_smooth + left_smooth

    if config.symmetry_mode == "asymmetric":
        skeleton_r = compute_skeleton(config.num_axes, config.distance_func, layer_seed, rng)
        skeleton_l = compute_skeleton(config.num_axes, config.distance_func, layer_seed + rng.random() * 50000, rng)
        right_pts = skeleton_to_points(skeleton_r, (cx, cy), max_radius, config.axes_angles)
        left_pts = skeleton_to_points(skeleton_l, (cx, cy), max_radius, config.axes_angles)
        # 镜像左半
        left_pts = [(2 * cx - x, y) for x, y in left_pts]
        right_smooth = draw_smooth_curve(right_pts)
        left_smooth = draw_smooth_curve(list(reversed(left_pts)))
        return right_smooth + left_smooth

    if config.symmetry_mode == "rotational":
        folds = max(config.rotational_folds, 2)
        half_wedge = max(config.num_axes // 2, 3)
        wedge_angles = np.linspace(0, 180.0 / folds, half_wedge).tolist()
        skeleton = compute_skeleton(half_wedge, config.distance_func, layer_seed, rng)
        right_pts = skeleton_to_points(skeleton, (cx, cy), max_radius, wedge_angles)
        right_smooth = draw_smooth_curve(right_pts)
        left_smooth = [(2 * cx - x, y) for x, y in reversed(right_smooth)]
        wedge = right_smooth + left_smooth

        all_points = []
        for fold in range(folds):
            angle = fold * 360.0 / folds
            all_points.extend(rotate_points(wedge, (cx, cy), angle))
        return all_points

    return []


# ============================================================
# 主流程
# ============================================================
def generate_and_save_images():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_rng = random.Random(RANDOM_SEED)

    for idx in range(NUM_IMAGES):
        image_rng = random.Random(RANDOM_SEED * 1000 + idx)
        config = generate_random_config(image_rng)
        img = draw_abstract_shape(IMAGE_SIZE, config, image_rng)
        path = os.path.join(OUTPUT_DIR, f"abstract_{idx:03d}.png")
        img.save(path)

        if (idx + 1) % 25 == 0:
            print(f"已生成 {idx + 1}/{NUM_IMAGES} 张图片")

    print(f"全部完成！图片保存在 {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_and_save_images()
