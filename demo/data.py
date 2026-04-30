from dataclasses import dataclass
from typing import *
from enum import Enum

class Color(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
class Shape(Enum):
    CUBE = "cube"
    SPHERE = "sphere"
    TETRAHEDRON = "tetrahedron"
    
@dataclass
class demo_object:
    color:Color
    shape:Shape
    def __str__(self):
        return f"{self.shape.value}_{self.color.value}"
    
all_objects = [demo_object(color, shape) for color in Color for shape in Shape]

