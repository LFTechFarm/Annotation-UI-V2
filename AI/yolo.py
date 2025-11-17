
from core.utils import yolo_to_pixels, pixels_to_yolo, calculate_iou
from core.models import BBox
import os
from tkinter import filedialog, messagebox


# ---------- YOLO integration ----------
def load_yolo_model(self):
    model_path = filedialog.askopenfilename(title="Select YOLO Model (e.g., best.pt)")
    if model_path:
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO(model_path)
            messagebox.showinfo("Success", "YOLO model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")

def predict_current_image(self):
    if not getattr(self, "yolo_model", None):
        messagebox.showwarning("Warning", "Load a YOLO model first.")
        return
    if not self.current_image:
        return
    try:
        results = self.yolo_model.predict(self.current_image, imgsz=max(self.img_w, self.img_h))
        img_path = self.image_files[self.current_index]
        fname = os.path.splitext(os.path.basename(img_path))[0]
        pred_txt_path = os.path.join(self.prediction_folder, fname + ".txt")
        bboxes = []
        for r in results:
            for box in r.boxes:
                xy = getattr(box, "xyxy", None)
                clsattr = getattr(box, "cls", None)
                if xy is None:
                    continue
                coords = xy[0].tolist() if hasattr(xy, "__len__") else [v for v in xy]
                if len(coords) >= 4:
                    x1, y1, x2, y2 = coords[:4]
                    cls = int(clsattr[0]) if (clsattr is not None and hasattr(clsattr, "__len__")) else int(getattr(box, "cls", 0))
                    bboxes.append(BBox(cls, x1, y1, x2, y2))
        self.bboxes_pred[img_path] = bboxes
        # save preds to file
        lines = []
        for bb in bboxes:
            nx, ny, nw, nh = bb.normalize(self.img_w, self.img_h)
            lines.append(f"{bb.cls} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")
        with open(pred_txt_path, "w") as f:
            f.write("\n".join(lines))
        messagebox.showinfo("Success", f"Predictions saved to:\n{pred_txt_path}")
        if hasattr(self, '_render_image_on_canvas'):
            self._render_image_on_canvas()
    except Exception as e:
        messagebox.showerror("Error", f"Prediction failed: {e}")

# ---------- Utility functions ----------
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
