def apply_mvp(v, mvp):
    v = mvp @ v
    return v / v[3]

def ndc_to_screen(v):
    x = (v[0] + 1) * 0.5
    y = (v[1] + 1) * 0.5
    return x, y

def transform_vertex(v, mvp):
    return ndc_to_screen(apply_mvp(v, mvp))