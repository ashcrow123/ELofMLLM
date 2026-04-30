import os
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")

OBJECTS = {
    "sphere": "球体",
    "cube": "立方体",
    "tetrahedron": "正三角体",
}
COLORS = {
    "red": "red",
    "blue": "blue",
    "green": "green",
}

IMAGE_COUNT = 10
IMAGE_SIZE = (4, 4)


def make_dirs():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    for object_name in OBJECTS:
        for color_name in COLORS:
            out_dir = os.path.join(IMAGES_DIR, f"{object_name}_{color_name}")
            os.makedirs(out_dir, exist_ok=True)


def draw_sphere(ax, color):
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, shade=True, linewidth=0, antialiased=False)


def draw_cube(ax, color):
    r = [-1, 1]
    vertices = [
        [x, y, z]
        for x in r
        for y in r
        for z in r
    ]
    faces = [
        [vertices[i] for i in [0, 1, 3, 2]],
        [vertices[i] for i in [4, 5, 7, 6]],
        [vertices[i] for i in [0, 1, 5, 4]],
        [vertices[i] for i in [2, 3, 7, 6]],
        [vertices[i] for i in [0, 2, 6, 4]],
        [vertices[i] for i in [1, 3, 7, 5]],
    ]
    collection = Poly3DCollection(faces, facecolors=color, linewidths=0.5, edgecolors="black", alpha=1)
    ax.add_collection3d(collection)


def draw_tetrahedron(ax, color):
    sqrt2 = np.sqrt(2)
    vertices = np.array([
        [1, 1, 1],
        [-1, -1, 1],
        [-1, 1, -1],
        [1, -1, -1],
    ]) / sqrt2
    faces = [
        [vertices[i] for i in [0, 1, 2]],
        [vertices[i] for i in [0, 1, 3]],
        [vertices[i] for i in [0, 2, 3]],
        [vertices[i] for i in [1, 2, 3]],
    ]
    collection = Poly3DCollection(faces, facecolors=color, linewidths=0.5, edgecolors="black", alpha=1)
    ax.add_collection3d(collection)


def configure_axes(ax):
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_zlim(-1.1, 1.1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.grid(False)


def save_random_views(object_name, draw_fn, color_name, color_value):
    out_dir = os.path.join(IMAGES_DIR, f"{object_name}_{color_name}")
    for index in range(1, IMAGE_COUNT + 1):
        fig = plt.figure(figsize=IMAGE_SIZE)
        ax = fig.add_subplot(111, projection="3d")
        configure_axes(ax)
        draw_fn(ax, color_value)
        elev = random.uniform(10, 80)
        azim = random.uniform(0, 360)
        ax.view_init(elev=elev, azim=azim)
        filename = os.path.join(out_dir, f"img_{index:02d}.png")
        plt.savefig(filename, dpi=150, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)


def main():
    make_dirs()
    random.seed(42)
    for object_name, object_label in OBJECTS.items():
        draw_fn = {
            "sphere": draw_sphere,
            "cube": draw_cube,
            "tetrahedron": draw_tetrahedron,
        }[object_name]
        for color_name, color_value in COLORS.items():
            save_random_views(object_name, draw_fn, color_name, color_value)
    print("Done: generated images in demo/images")


if __name__ == "__main__":
    main()
