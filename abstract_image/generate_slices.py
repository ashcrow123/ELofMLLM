"""生成对称兰花形状图片"""

import numpy as np
from dataclasses import dataclass
from PIL import Image, ImageDraw
from scipy.interpolate import make_interp_spline
import os

# 配置
NUM_AXES = 7  # 可自定义的轴数量
NUM_IMAGES = 100
IMAGE_SIZE = 256
ORCHID_COLOR = (219, 112, 147)
BG_COLOR = (255, 255, 255)
RANDOM_SEED = 44
ANGULAR_FREQ = 1 / 400  # 角频率
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "images")
# 根据轴数量等分角度（0到180度）
ANGLES = np.linspace(0, 180, NUM_AXES)


@dataclass
class Skeleton:
    """骨架数据：轴上点到端点的距离"""
    distances: list[float]

    def __post_init__(self):
        self.distances = list(self.distances)
    @property
    def num_axes(self):
        return len(self.distances)

def generate_phases(seed: int, num_axes: int) -> np.ndarray:
    """使用随机种子生成各轴的初始相位"""
    rng = np.random.default_rng(seed)
    return rng.random(num_axes) * 2 * np.pi  # 相位范围 [0, 2π]


def compute_skeleton(image_idx: int, phases: np.ndarray) -> Skeleton:
    """
    计算第image_idx张图的骨架数据

    距离 = cos(i * angular_freq * 2π + phase)
    映射到 [0.1, 1.0] 范围（留边距避免贴着边界）
    """
    # 归一化的角频率：2π/300
    omega = 2 * np.pi * ANGULAR_FREQ
    angles_rad = np.deg2rad(ANGLES)

    distances = []
    for angle_idx in range(len(ANGLES)):
        value = np.cos(image_idx * omega * (np.log(angle_idx+1)+1)+ phases[angle_idx])
        value = abs(value)  # 取绝对值
        # 将 [0, 1] 映射到 [0.1, 0.9]（留白
        # 边）
        distance = 0.1 + 0.8 * value
        distances.append(distance)

    return Skeleton(distances=distances)


def draw_smooth_curve(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """
    使用三次样条插值生成平滑曲线上的点

    Args:
        points: 原始数据点 [(x0,y0), (x1,y1), ...]

    Returns:
        插值后的平滑点列表
    """
    n = len(points)
    if n < 2:
        return points
    if n == 2:
        # 线性插值
        x0, y0 = points[0]
        x1, y1 = points[1]
        result = []
        for i in range(100):
            t = i / 99
            result.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
        return result

    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    t = np.linspace(0, 1, n)
    t_new = np.linspace(0, 1, 200)

    spline = make_interp_spline(t, np.column_stack([x, y]), k=3, bc_type='natural')
    smooth_points = spline(t_new)

    return [(float(p[0]), float(p[1])) for p in smooth_points]


def skeleton_to_points(skeleton: Skeleton, center: tuple[float, float], max_radius: float) -> list[tuple[float, float]]:
    """
    将骨架数据转换为图像上的点坐标（右半边）

    ANGLES = [0, 30, 60, 90, 120, 150, 180]
    0度在上（顶点），180度在下（底部端点）
    """
    cx, cy = center
    points = []

    for i, angle in enumerate(ANGLES):
        angle_rad = np.deg2rad(angle)
        # 图像坐标系：x右，y下。0度在上，180度在下
        # r * cos(angle) = x偏移, r * sin(angle) = y偏移（向下为正）
        r = skeleton.distances[i] * max_radius
        x = cx + r * np.sin(angle_rad)  # sin(0)=0在顶部，sin(90)=1在右边
        y = cy - r * np.cos(angle_rad)   # cos(0)=1向上，cos(90)=0在中间

        points.append((x, y))

    return points


def draw_orchid_shape(size: int, skeleton: Skeleton) -> Image:
    """绘制单张兰花形状图片"""
    img = Image.new("RGB", (size, size), BG_COLOR)
    draw = ImageDraw.Draw(img)

    center = (size / 2, size / 2)
    max_radius = size / 2 * 0.85

    # 右半边7个点：0度在上，30，60，90，120，150，180在下
    right_points = skeleton_to_points(skeleton, center, max_radius)
    # right_points = [点0(0°), 点1(30°), 点2(60°), 点3(90°), 点4(120°), 点5(150°), 点6(180°)]
    # 从上到下依次排列

    # 生成右半边平滑曲线
    right_smooth = draw_smooth_curve(right_points)

    # 左半边是右半边的镜像
    left_points = []
    for x, y in right_points:
        left_points.append((size - x, y))
    left_points = list(reversed(left_points))  # 从下往上
    left_smooth = draw_smooth_curve(left_points)

    # 构建完整闭合多边形：右半边从上到下 + 左半边从下到上
    full_polygon = right_smooth + left_smooth

    # 填充颜色
    draw.polygon(full_polygon, fill=ORCHID_COLOR)

    return img


def generate_and_save_images():
    """生成并保存所有图片"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 生成初始相位
    phases = generate_phases(RANDOM_SEED, len(ANGLES))

    # 先生成所有骨架数据
    skeletons = [compute_skeleton(i, phases) for i in range(NUM_IMAGES)]

    for idx, skeleton in enumerate(skeletons):
        img = draw_orchid_shape(IMAGE_SIZE, skeleton)
        path = os.path.join(OUTPUT_DIR, f"orchid_{idx:03d}.png")
        img.save(path)

        if (idx + 1) % 50 == 0:
            print(f"已生成 {idx + 1}/{NUM_IMAGES} 张图片")

    print(f"全部完成！图片保存在 {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_and_save_images()