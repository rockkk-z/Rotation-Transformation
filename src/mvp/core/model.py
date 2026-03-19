import taichi as ti
import math

def get_model_matrix(angle):
    rad = angle * math.pi / 180.0

    return ti.Matrix([
        [ math.cos(rad), 0, math.sin(rad), 0],
        [ 0,             1, 0,             0],
        [-math.sin(rad), 0, math.cos(rad), 0],
        [ 0,             0, 0,             1]
    ])