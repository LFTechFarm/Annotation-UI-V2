
from PIL import Image, ImageFilter
import tkinter as tk
from tkinter import messagebox
from core.models import BBox


# ---------- Extra tool: Excessive Green ----------
def _compute_excessive_green_mask(self, pil_image, thresh):
    """Return a binary mask (mode 'L') where G - max(R,B) > thresh."""
    try:
        im = pil_image.convert("RGB")
        w, h = im.size
        r,g,b = im.split()
        # convert to lists to compute difference quickly
        r_px = r.load(); g_px = g.load(); b_px = b.load()
        mask = Image.new("L", (w,h))
        m_px = mask.load()
        for yy in range(h):
            for xx in range(w):
                gv = g_px[xx,yy]
                mv = max(r_px[xx,yy], b_px[xx,yy])
                if (gv - mv) > thresh:
                    m_px[xx,yy] = 255
                else:
                    m_px[xx,yy] = 0
        # optionally filter small speckles
        mask = mask.filter(ImageFilter.MaxFilter(3))
        return mask
    except Exception as e:
        print("Mask error:", e)
        return None

def excessive_green_apply(self):
    """Apply excessive green -> find bounding boxes from mask and add them to extras."""
    if not self.current_image:
        return
    thresh = getattr(self, "eg_thresh_var", tk.IntVar(value=50)).get()
    mask = _compute_excessive_green_mask(self, self.current_image, thresh)
    if mask is None:
        messagebox.showerror("Error", "Failed to compute mask.")
        return
    # find bounding boxes of mask regions — for simplicity, take bounding box of entire mask
    bbox = mask.getbbox()  # returns (left, upper, right, lower) or None
    if not bbox:
        messagebox.showinfo("No mask", "Mask produced no region.")
        return
        # map from mask (original image coords) to bbox coords
    x1, y1, x2, y2 = bbox
    new_bb = BBox(0, x1, y1, x2, y2)
    img_path = self.image_files[self.current_index]
    self.bboxes_extra.setdefault(img_path, []).append(new_bb)
    messagebox.showinfo("Extra Added", "Added a bbox from Excessive Green mask to Extras.")
    self._render_image_on_canvas()



