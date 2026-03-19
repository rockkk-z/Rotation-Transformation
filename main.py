# -*- coding: utf-8 -*-

import taichi as ti
import math

ti.init(arch=ti.cpu)

width, height = 700, 700
gui = ti.GUI("MVP Transform (Final)", res=(width, height))


# 数据
triangle_vertices = ti.Vector.field(3, dtype=ti.f32, shape=3)
triangle_screen = ti.Vector.field(2, dtype=ti.f32, shape=3)

cube_vertices = ti.Vector.field(3, dtype=ti.f32, shape=8)
cube_screen = ti.Vector.field(2, dtype=ti.f32, shape=8)

# 初始化三角形
triangle_vertices[0] = [2.0, 0.0, -2.0]
triangle_vertices[1] = [0.0, 2.0, -2.0]
triangle_vertices[2] = [-2.0, 0.0, -2.0]

# 初始化立方体
cube_data = [
    [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
    [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]
]
for i in range(8):
    cube_vertices[i] = cube_data[i]


# 变换矩阵

# 立方体（绕Y轴）
@ti.func
def get_model_matrix_y(angle):
    rad = angle * math.pi / 180.0
    c = ti.cos(rad)
    s = ti.sin(rad)

    return ti.Matrix([
        [c, 0.0, s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

# 三角形（绕Z轴）
@ti.func
def get_model_matrix_z(angle):
    rad = angle * math.pi / 180.0
    c = ti.cos(rad)
    s = ti.sin(rad)

    return ti.Matrix([
        [c, -s, 0.0, 0.0],
        [s,  c, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_view_matrix():
    return ti.Matrix([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, -5.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

@ti.func
def get_projection_matrix():
    fov = 45.0 * math.pi / 180.0
    n = -0.1
    f = -50.0

    t = ti.tan(fov / 2.0) * ti.abs(n)
    r = t
    l = -r
    b = -t

    M_p2o = ti.Matrix([
        [n, 0.0, 0.0, 0.0],
        [0.0, n, 0.0, 0.0],
        [0.0, 0.0, n + f, -n * f],
        [0.0, 0.0, 1.0, 0.0]
    ])

    M_scale = ti.Matrix([
        [2/(r-l),0,0,0],
        [0,2/(t-b),0,0],
        [0,0,2/(n-f),0],
        [0,0,0,1]
    ])

    M_trans = ti.Matrix([
        [1,0,0,-(r+l)/2],
        [0,1,0,-(t+b)/2],
        [0,0,1,-(n+f)/2],
        [0,0,0,1]
    ])

    return M_scale @ M_trans @ M_p2o


@ti.kernel
def compute(angle: ti.f32):
    view = get_view_matrix()
    proj = get_projection_matrix()

    # 三角形
    model_tri = get_model_matrix_z(angle)
    mvp_tri = proj @ view @ model_tri

    for i in range(3):
        v = triangle_vertices[i]
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])

        v_clip = mvp_tri @ v4
        v_ndc = v_clip / v_clip[3]

        triangle_screen[i] = [(v_ndc[0]+1)/2, (v_ndc[1]+1)/2]

    # 立方体
    model_cube = get_model_matrix_y(angle)
    mvp_cube = proj @ view @ model_cube

    for i in range(8):
        v = cube_vertices[i]
        v4 = ti.Vector([v[0], v[1], v[2], 1.0])

        v_clip = mvp_cube @ v4
        v_ndc = v_clip / v_clip[3]

        cube_screen[i] = [(v_ndc[0]+1)/2, (v_ndc[1]+1)/2]


# 边定义
triangle_edges = [(0,1),(1,2),(2,0)]

cube_edges = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7)
]

cube_colors = [
    0xFF0000,0x00FF00,0x0000FF,0xFFFF00,
    0xFF00FF,0x00FFFF,0xFFFFFF,0xFFA500,
    0x800080,0x008000,0x000080,0x808000
]


# 主循环
angle = 0.0
mode = "cube"

while gui.running:
    if gui.get_event(ti.GUI.PRESS):
        if gui.event.key == 'a':
            angle += 10.0
        elif gui.event.key == 'd':
            angle -= 10.0
        elif gui.event.key == 't':
            mode = "triangle"
        elif gui.event.key == 'c':
            mode = "cube"
        elif gui.event.key == ti.GUI.ESCAPE:
            break

    compute(angle)

    gui.clear(0x000000)

    # 三角形
    if mode == "triangle":
        gui.line(triangle_screen[0], triangle_screen[1], radius=2, color=0xFF0000)
        gui.line(triangle_screen[1], triangle_screen[2], radius=2, color=0x00FF00)
        gui.line(triangle_screen[2], triangle_screen[0], radius=2, color=0x0000FF)

    # 立方体
    else:
        for i, e in enumerate(cube_edges):
            gui.line(cube_screen[e[0]], cube_screen[e[1]], radius=2, color=cube_colors[i])

    gui.text(f"Mode: {mode}", pos=(0.02, 0.95), color=0xFFFFFF)
    gui.show()