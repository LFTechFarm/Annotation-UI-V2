def yolo_to_pixels(nx, ny, nw, nh, w, h):
    cx = nx * w
    cy = ny * h
    bw = nw * w
    bh = nh * h
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2
    return x1, y1, x2, y2

def pixels_to_yolo(x1, y1, x2, y2, w, h):
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = x1 + bw / 2
    cy = y1 + bh / 2
    return cx / w, cy / h, bw / w, bh / h

def calculate_iou(b1, b2):
    x11, y11, x12, y12 = b1
    x21, y21, x22, y22 = b2
    ix1, iy1 = max(x11, x21), max(y11, y21)
    ix2, iy2 = min(x12, x22), min(y12, y22)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a1 = max(0, (x12 - x11)) * max(0, (y12 - y11))
    a2 = max(0, (x22 - x21)) * max(0, (y22 - y21))
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0
