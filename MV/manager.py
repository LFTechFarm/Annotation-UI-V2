"""Simple contour-based rectangle/square detection for Machine Vision panel.

Provides `rectangle_detect(self)` which reads the current image file (like Circle.py),
finds rectangular contours, draws them on a preview image and adds them as extras
in `self.bboxes_extra`.
"""
from PIL import Image
def rectangle_detect(self):
    try:
        import cv2
        import numpy as np
    except Exception as e:
        print("OpenCV is required for rectangle detection:", e)
        return

    # find current image path similar to Circle helper
    path = None
    try:
        if getattr(self, "image_files", None) and getattr(self, "current_index", -1) >= 0:
            path = self.image_files[self.current_index]
    except Exception:
        path = None
    if not path:
        path = getattr(self, "current_image_path", None)
    if not path:
        print("No image loaded for rectangle detection.")
        return

    img = cv2.imread(path)
    if img is None:
        print("Could not read image:", path)
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    # read parameters from UI if available
    try:
        low = int(self.rect_canny_low.get()) if getattr(self, 'rect_canny_low', None) is not None else 50
    except Exception:
        low = 50
    try:
        high = int(self.rect_canny_high.get()) if getattr(self, 'rect_canny_high', None) is not None else 150
    except Exception:
        high = 150
    try:
        eps = float(self.rect_approx_eps.get()) if getattr(self, 'rect_approx_eps', None) is not None else 0.02
    except Exception:
        eps = 0.02
    try:
        min_area = int(self.rect_min_area.get()) if getattr(self, 'rect_min_area', None) is not None else 100
    except Exception:
        min_area = 100

    edged = cv2.Canny(gray, low, high)

    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    rects = []
    vis = img.copy()
    for cnt in contours:
        # approximate contour
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            area = w * h
            if area < min_area:  # skip tiny boxes
                continue
            # optional aspect ratio checks can be applied if needed
            # enforce approx epsilon by re-approximating with provided eps fraction
            # (eps fraction of perimeter)
            # already approximated above using 0.02; recompute with requested eps for consistency
            peri = cv2.arcLength(cnt, True)
            approx2 = cv2.approxPolyDP(cnt, eps * peri, True)
            if len(approx2) != 4:
                continue
            rects.append((x, y, x + w, y + h))
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # save preview and store detected rects (do NOT auto-add to extras)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    self.mv_preview_image = Image.fromarray(vis_rgb)
    # store detected rect list for potential conversion
    try:
        self.detected_rects = rects
    except Exception:
        self.detected_rects = list(rects)

    # render preview
    if hasattr(self, '_render_image_on_canvas'):
        self._render_image_on_canvas()

def circle_detect(self):
    """Detect circles using HoughCircles with parameters from the UI (if present)."""
    try:
        import cv2
        import numpy as np
    except Exception as e:
        print("OpenCV is required for circle detection:", e)
        return
    from tkinter import messagebox

    # get path
    path = None
    try:
        if getattr(self, "image_files", None) and getattr(self, "current_index", -1) >= 0:
            path = self.image_files[self.current_index]
    except Exception:
        path = None
    if not path:
        path = getattr(self, "current_image_path", None)
    if not path:
        print("No image loaded for circle detection.")
        return

    img_bgr = cv2.imread(path)
    if img_bgr is None:
        print("Could not read image:", path)
        return

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # read parameters from UI variables if present, else defaults
    p1 = getattr(self, 'hough_param1', None)
    p2 = getattr(self, 'hough_param2', None)
    min_r = getattr(self, 'hough_min_r', None)
    max_r = getattr(self, 'hough_max_r', None)
    minDist = getattr(self, 'hough_minDist', None)
    try:
        p1v = int(p1.get()) if p1 is not None else 100
        p2v = int(p2.get()) if p2 is not None else 30
        minrv = int(min_r.get()) if min_r is not None else 10
        maxrv = int(max_r.get()) if max_r is not None else 80
        mindv = int(minDist.get()) if minDist is not None else 20
    except Exception:
        p1v, p2v, minrv, maxrv, mindv = 100, 30, 10, 80, 20

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=mindv,
        param1=p1v,
        param2=p2v,
        minRadius=minrv,
        maxRadius=maxrv
    )

    # clear previous circles, do not auto-add to extras
    vis_img = img_bgr.copy()
    found = 0
    self.detected_circles = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for (x, y, r) in circles[0, :]:
            cv2.circle(vis_img, (x, y), r, (0, 255, 0), 2)
            cv2.circle(vis_img, (x, y), 2, (0, 0, 255), 3)
            self.detected_circles.append((x, y, r))
            found += 1

    # store preview and detected circles only (no auto-append)
    self.mv_preview_image = vis_img
    if hasattr(self, '_render_image_on_canvas'):
        self._render_image_on_canvas()

def triangle_detect(self):
    """Detect triangular shapes by contour approx (len==3)."""
    try:
        import cv2
        import numpy as np
    except Exception as e:
        print("OpenCV required:", e)
        return

    # get image path
    path = None
    try:
        if getattr(self, "image_files", None) and getattr(self, "current_index", -1) >= 0:
            path = self.image_files[self.current_index]
    except Exception:
        path = None
    if not path:
        path = getattr(self, "current_image_path", None)
    if not path:
        print("No image loaded for triangle detection.")
        return

    img = cv2.imread(path)
    if img is None:
        print("Could not read image:", path)
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    tris = []
    vis = img.copy()

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) == 3:
            x, y, w, h = cv2.boundingRect(approx)
            area = w * h
            if area < 50:
                continue
            tris.append(approx.reshape(-1, 2).tolist())  # store triangle points
            cv2.drawContours(vis, [approx], -1, (255, 0, 0), 2)

    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    self.mv_preview_image = Image.fromarray(vis_rgb)
    self.detected_triangles = tris
    if hasattr(self, '_render_image_on_canvas'):
        self._render_image_on_canvas()


def polygon_detect(self):
    """Detect polygons with N vertices (uses contour approximation)."""
    try:
        import cv2
        import numpy as np
    except Exception as e:
        print("OpenCV required:", e)
        return

    # get image path
    path = None
    try:
        if getattr(self, "image_files", None) and getattr(self, "current_index", -1) >= 0:
            path = self.image_files[self.current_index]
    except Exception:
        path = None
    if not path:
        path = getattr(self, "current_image_path", None)
    if not path:
        print("No image loaded for polygon detection.")
        return

    img = cv2.imread(path)
    if img is None:
        print("Could not read image:", path)
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    # parameters from UI
    try:
        low = int(self.rect_canny_low.get()) if getattr(self, 'rect_canny_low', None) else 50
    except Exception:
        low = 50
    try:
        high = int(self.rect_canny_high.get()) if getattr(self, 'rect_canny_high', None) else 150
    except Exception:
        high = 150
    try:
        min_area = int(self.rect_min_area.get()) if getattr(self, 'rect_min_area', None) else 100
    except Exception:
        min_area = 100
    try:
        n = int(self.poly_n.get()) if getattr(self, 'poly_n', None) else 5
    except Exception:
        n = 5

    edged = cv2.Canny(gray, low, high)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polys = []
    vis = img.copy()

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == n and cv2.isContourConvex(approx):
            x, y, w, h = cv2.boundingRect(approx)
            area = w * h
            if area < min_area:
                continue
            polys.append(approx.reshape(-1, 2).tolist())  # store polygon points
            cv2.drawContours(vis, [approx], -1, (0, 180, 255), 2)

    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    self.mv_preview_image = Image.fromarray(vis_rgb)
    self.detected_polygons = polys

    img_path = self.image_files[self.current_index] if getattr(self, 'image_files', None) else None
    if img_path and polys:
        try:
            from core.models import BBox
            for contour in polys:
                x_vals, y_vals = zip(*contour)
                self.bboxes_extra.setdefault(img_path, []).append(
                    BBox(0, min(x_vals), min(y_vals), max(x_vals), max(y_vals))
                )
        except Exception:
            pass

    if hasattr(self, '_render_image_on_canvas'):
        self._render_image_on_canvas()
