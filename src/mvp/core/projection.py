import taichi as ti
import math

def get_projection_matrix(eye_fov, aspect_ratio, zNear, zFar):
    fov = eye_fov * math.pi / 180.0

    n = -zNear
    f = -zFar

    t = math.tan(fov / 2) * abs(n)
    b = -t
    r = aspect_ratio * t
    l = -r

    persp_to_ortho = ti.Matrix([
        [n, 0, 0, 0],
        [0, n, 0, 0],
        [0, 0, n + f, -n * f],
        [0, 0, 1, 0]
    ])

    ortho_translate = ti.Matrix([
        [1, 0, 0, -(r + l) / 2],
        [0, 1, 0, -(t + b) / 2],
        [0, 0, 1, -(n + f) / 2],
        [0, 0, 0, 1]
    ])

    ortho_scale = ti.Matrix([
        [2 / (r - l), 0, 0, 0],
        [0, 2 / (t - b), 0, 0],
        [0, 0, 2 / (n - f), 0],
        [0, 0, 0, 1]
    ])

    return ortho_scale @ ortho_translate @ persp_to_ortho