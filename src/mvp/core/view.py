import taichi as ti

def get_view_matrix(eye_pos):
    ex, ey, ez = eye_pos

    return ti.Matrix([
        [1, 0, 0, -ex],
        [0, 1, 0, -ey],
        [0, 0, 1, -ez],
        [0, 0, 0, 1]
    ])