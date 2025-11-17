from .utils import pixels_to_yolo

class BBox:
    def __init__(self, cls: int, x1: float, y1: float, x2: float, y2: float, alpha: float = 1.0):
        self.cls = int(cls)
        self.x1 = float(x1)
        self.y1 = float(y1)
        self.x2 = float(x2)
        self.y2 = float(y2)
        self.alpha = float(alpha)
    def as_tuple(self):
        return (self.x1, self.y1, self.x2, self.y2)
    def normalize(self, img_w, img_h):
        return pixels_to_yolo(self.x1, self.y1, self.x2, self.y2, img_w, img_h)
