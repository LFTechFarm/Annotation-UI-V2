
"""
Unified shape detection functions for the Machine Vision panel.
Each function works directly with the latest in-memory image (RGB / Mask / Index).
"""

from PIL import Image
import numpy as np
import cv2



# ================================================================
# 🔵 Circle Detection

def circle_detect(self, pil_image):
    """
    Detect circles using OpenCV's HoughCircles.
    Results are stored in self.mv_detections["circles"] as (x, y, r) tuples.
    Bounding boxes are created later via _mv_convert_detected("detected_circles").
    """
    import cv2
    import numpy as np
    from PIL import Image

    try:
        # --- Clear stale bbox results before new detection ---
        if "circles_bboxes" in getattr(self, "mv_detections", {}):
            print("[DEBUG] Clearing old circles_bboxes before detection.")
            self.mv_detections.pop("circles_bboxes", None)

        # --- Read parameters from GUI sliders (with defaults) ---
        dp = 1.0
        min_dist = int(self.hough_minDist.get()) if hasattr(self, "hough_minDist") else 20
        param1 = int(self.hough_param1.get()) if hasattr(self, "hough_param1") else 100
        param2 = int(self.hough_param2.get()) if hasattr(self, "hough_param2") else 30
        min_r  = int(self.hough_min_r.get())   if hasattr(self, "hough_min_r")   else 5
        max_r  = int(self.hough_max_r.get())   if hasattr(self, "hough_max_r")   else 80

        print(f"[DEBUG] Hough params: minDist={min_dist}, param1={param1}, "
              f"param2={param2}, min_r={min_r}, max_r={max_r}")

        # --- Convert PIL → OpenCV grayscale image ---
        img_rgb = np.array(pil_image.convert("RGB"))
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        gray_blur = cv2.medianBlur(gray, 5)

        # --- Run Hough Circle Transform ---
        circles = cv2.HoughCircles(
            gray_blur,
            cv2.HOUGH_GRADIENT,
            dp=dp,
            minDist=min_dist,
            param1=param1,
            param2=param2,
            minRadius=min_r,
            maxRadius=max_r,
        )

        detected = []
        vis = cv2.cvtColor(gray_blur, cv2.COLOR_GRAY2BGR)

        if circles is not None:
            circles = np.uint16(np.around(circles[0, :]))
            for x, y, r in circles:
                detected.append((int(x), int(y), int(r)))
                # draw for preview
                cv2.circle(vis, (x, y), r, (0, 255, 0), 2)
                cv2.circle(vis, (x, y), 2, (0, 0, 255), 3)

        # --- Store results ---
        self.mv_detections["circles"] = detected
        self.detected_circles = detected

        # Create preview image for the GUI canvas
        self.mv_preview_image = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        self._render_image_on_canvas()

        print(f"[INFO] {len(detected)} circle(s) detected with HoughCircles.")
        print(f"[DEBUG] Stored {len(detected)} circles in mv_detections['circles']")
        print(f"[DEBUG] After detection, mv_detections keys: {list(self.mv_detections.keys())}")

    except Exception as e:
        print(f"[ERROR] circle_detect failed: {e}")
def circle_detect_circularity(self, pil_image):
    """
    Detect circular blobs using contour circularity measure.
    Results stored in:
        self.mv_detections["circularity"]
        self.mv_preview_image (for rendering)
    """
    import cv2
    import numpy as np
    from PIL import Image

    try:
        img_rgb = np.array(pil_image.convert("RGB"))
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (5, 5), 1.5)

        # Threshold to binary
        _, thr = cv2.threshold(gray_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        vis = cv2.cvtColor(gray_blur, cv2.COLOR_GRAY2BGR)
        detected = []

        # Read slider parameters
        circ_thr = float(self.circularity_threshold.get()) if hasattr(self, "circularity_threshold") else 0.7
        min_area = int(self.circularity_min_area.get()) if hasattr(self, "circularity_min_area") else 30
        min_r = int(self.hough_min_r.get()) if hasattr(self, "hough_min_r") else 5
        max_r = int(self.hough_max_r.get()) if hasattr(self, "hough_max_r") else 80

        print(f"[DEBUG] Circularity params: thr={circ_thr:.2f}, min_area={min_area}, r=({min_r}-{max_r})")

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            peri = cv2.arcLength(cnt, True)
            if peri <= 0:
                continue

            circularity = 4 * np.pi * (area / (peri * peri))
            if circularity < circ_thr:
                continue

            (x, y), r = cv2.minEnclosingCircle(cnt)
            x, y, r = int(x), int(y), int(r)
            if r < min_r or r > max_r:
                continue

            detected.append((x, y, r))
            cv2.circle(vis, (x, y), r, (255, 0, 0), 2)   # Blue circle
            cv2.putText(vis, f"{circularity:.2f}", (x - 20, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        self.mv_detections["circularity"] = detected
        self.detected_circularity = detected

        # ✅ Important: ensure a valid image preview is stored
        self.mv_preview_image = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        self._render_image_on_canvas()

        print(f"[INFO] {len(detected)} circular blob(s) detected by circularity filter.")
        print(f"[DEBUG] Stored {len(detected)} in mv_detections['circularity']")

    except Exception as e:
        print(f"[ERROR] circle_detect_circularity failed: {e}")

 
# ================================================================
# 🟩 Rectangle / Square Detection
# ================================================================

def rectangle_detect(app, pil_image):
    """
    Detect rectangular shapes (squares/rectangles) using contours.
    Updates app.detected_rects and app.mv_preview_image.
    """
    try:
        img = np.array(pil_image.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Paramètres UI
        low = app.rect_canny_low.get() if hasattr(app, "rect_canny_low") else 50
        high = app.rect_canny_high.get() if hasattr(app, "rect_canny_high") else 150
        min_area = app.rect_min_area.get() if hasattr(app, "rect_min_area") else 100
        eps = app.rect_approx_eps.get() if hasattr(app, "rect_approx_eps") else 0.02

        # Détection de contours
        edges = cv2.Canny(gray, int(low), int(high))
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected = []
        vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, eps * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                x, y, w, h = cv2.boundingRect(approx)
                detected.append((x, y, x + w, y + h))
                cv2.drawContours(vis, [approx], -1, (0, 255, 0), 2)

        app.detected_rects = detected
        app.mv_preview_image = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        print(f"[INFO] Detected {len(detected)} rectangles.")
        app._render_image_on_canvas()

    except Exception as e:
        print(f"[ERROR] rectangle_detect failed: {e}")

# ================================================================
# 🔺 Triangle Detection
# ================================================================
import cv2
import numpy as np
from PIL import Image

def triangle_detect(app, pil_image):
    """
    Detect triangular shapes using contour approximation.
    Updates app.detected_triangles and app.mv_preview_image.
    """
    try:
        img = np.array(pil_image.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Parameters
        low = app.rect_canny_low.get() if hasattr(app, "rect_canny_low") else 50
        high = app.rect_canny_high.get() if hasattr(app, "rect_canny_high") else 150
        min_area = app.rect_min_area.get() if hasattr(app, "rect_min_area") else 100
        eps = app.rect_approx_eps.get() if hasattr(app, "rect_approx_eps") else 0.02

        edges = cv2.Canny(blur, int(low), int(high))
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        detected = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, eps * peri, True)

            if len(approx) == 3 and cv2.isContourConvex(approx):
                x, y, w, h = cv2.boundingRect(approx)
                detected.append((x, y, x + w, y + h))
                cv2.drawContours(vis, [approx], -1, (255, 255, 0), 2)

        app.detected_triangles = detected
        app.mv_preview_image = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        print(f"[INFO] Detected {len(detected)} triangles.")
        app._render_image_on_canvas()

    except Exception as e:
        print(f"[ERROR] triangle_detect failed: {e}")


# ================================================================
# ⬣ Polygon Detection
# ================================================================
import cv2
import numpy as np
from PIL import Image

def polygon_detect(app, pil_image):
    """
    Detect N-sided polygons in the input image.
    Updates app.detected_polygons and app.mv_preview_image.
    """
    try:
        n_sides = int(app.poly_n.get()) if hasattr(app, "poly_n") else 4

        img = np.array(pil_image.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Params: reuse same as rect detection
        low = app.rect_canny_low.get() if hasattr(app, "rect_canny_low") else 50
        high = app.rect_canny_high.get() if hasattr(app, "rect_canny_high") else 150
        min_area = app.rect_min_area.get() if hasattr(app, "rect_min_area") else 100
        eps = app.rect_approx_eps.get() if hasattr(app, "rect_approx_eps") else 0.02

        edges = cv2.Canny(blur, int(low), int(high))
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        vis = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        detected = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < min_area:
                continue

            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, eps * peri, True)

            if len(approx) == n_sides and cv2.isContourConvex(approx):
                pts = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]
                detected.append(pts)
                cv2.drawContours(vis, [approx], -1, (255, 128, 0), 2)

        app.detected_polygons = detected
        app.mv_preview_image = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        print(f"[INFO] Detected {len(detected)} polygons (n={n_sides}).")

        app._render_image_on_canvas()

    except Exception as e:
        print(f"[ERROR] polygon_detect failed: {e}")


def contour_detect_generic(app, pil_image):
    """
    Robust contour detection for binary masks or RGB images.
    """
    try:
        print("[DEBUG] Running generic contour detection...")

        img = np.array(pil_image.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # --- Normalize and binarize ---
        _, bin_mask = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

        # --- Optional: skip Canny for clean masks ---
        use_canny = np.mean(bin_mask) < 250  # heuristic
        if use_canny:
            low = app.rect_canny_low.get() if hasattr(app, "rect_canny_low") else 50
            high = app.rect_canny_high.get() if hasattr(app, "rect_canny_high") else 150
            edges = cv2.Canny(bin_mask, int(low), int(high))
        else:
            edges = bin_mask.copy()

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        detected = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < getattr(app, "rect_min_area", 100):
                continue

            eps = 0.01 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, eps, True)
            pts = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]
            detected.append(pts)
            cv2.drawContours(vis, [approx], -1, (0, 255, 255), 2)

        app.detected_contours = detected
        app.mv_preview_image = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        print(f"[INFO] Detected {len(detected)} contours.")
        app._render_image_on_canvas()

    except Exception as e:
        print(f"[ERROR] contour_detect failed: {e}")
def contour_detect_mask(app, pil_image):
    """
    Detect outer contours of white regions in a binary mask (mode L or RGB).
    Returns list of point lists [(x, y), ...] for each contour.
    """
    try:
        print("[DEBUG] Running generic contour detection (mask mode)...")

        # --- Convert to binary OpenCV mask ---
        img = np.array(pil_image.convert("L"))  # grayscale
        _, mask = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)  # pure 0/255

        # --- Find contours directly on white regions ---
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        detected = []

        for c in contours:
            if cv2.contourArea(c) < 5:  # skip noise
                continue

            pts = [(int(pt[0][0]), int(pt[0][1])) for pt in c]
            detected.append(pts)
            cv2.drawContours(vis, [c], -1, (0, 255, 255), 2)

        app.detected_contours = detected
        app.mv_preview_image = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))

        print(f"[INFO] Detected {len(detected)} mask contours.")
        app._render_image_on_canvas()

    except Exception as e:
        print(f"[ERROR] contour_detect failed: {e}")


def detect_shape_generic(app, pil_image, shape_type):
    """
    Unified shape detection entrypoint.
    shape_type: 'circles', 'rects', 'triangles', 'polygons', 'contours'
    """
    try:
        shape = shape_type.lower()
        img = np.array(pil_image.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # --- Common params ---
        low = getattr(app, "rect_canny_low", None)
        high = getattr(app, "rect_canny_high", None)
        min_area = getattr(app, "rect_min_area", None)
        eps = getattr(app, "rect_approx_eps", None)

        low = int(low.get() if low else 50)
        high = int(high.get() if high else 150)
        min_area = int(min_area.get() if min_area else 100)
        eps = float(eps.get() if eps else 0.02)

        vis = img.copy()
        detected = []

        # ------------------------------------------------------------
        # HOUGH CIRCLES
        # ------------------------------------------------------------
        if shape == "circles":
            param1 = int(app.hough_param1.get())
            param2 = int(app.hough_param2.get())
            min_r = int(app.hough_min_r.get())
            max_r = int(app.hough_max_r.get())
            minDist = int(app.hough_minDist.get())

            circles = cv2.HoughCircles(
                blur, cv2.HOUGH_GRADIENT, dp=1,
                minDist=minDist, param1=param1, param2=param2,
                minRadius=min_r, maxRadius=max_r
            )

            if circles is not None:
                circles = np.uint16(np.around(circles[0, :]))
                for x, y, r in circles:
                    detected.append((x, y, r))
                    cv2.circle(vis, (x, y), r, (0, 0, 255), 2)

            app.detected_circles = detected

        # ------------------------------------------------------------
        # RECT / TRIANGLE / POLYGON / CONTOUR
        # ------------------------------------------------------------
        else:
            edges = cv2.Canny(blur, low, high)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            n_sides = int(app.poly_n.get()) if hasattr(app, "poly_n") else 5

            for c in contours:
                area = cv2.contourArea(c)
                if area < min_area:
                    continue
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, eps * peri, True)
                pts = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]

                if shape == "rects" and len(approx) == 4:
                    app.detected_rects = app.detected_rects + [cv2.boundingRect(c)] if hasattr(app, "detected_rects") else [cv2.boundingRect(c)]
                    cv2.drawContours(vis, [approx], -1, (0, 255, 0), 2)
                elif shape == "triangles" and len(approx) == 3:
                    detected.append(pts)
                    cv2.drawContours(vis, [approx], -1, (255, 255, 0), 2)
                elif shape == "polygons" and len(approx) == n_sides:
                    detected.append(pts)
                    cv2.drawContours(vis, [approx], -1, (255, 128, 0), 2)
                elif shape == "contours":
                    detected.append(pts)
                    cv2.drawContours(vis, [approx], -1, (0, 255, 255), 1)

            if shape == "rects":
                app.detected_rects = detected
            elif shape == "triangles":
                app.detected_triangles = detected
            elif shape == "polygons":
                app.detected_polygons = detected
            elif shape == "contours":
                app.detected_contours = detected

        # --- Update preview ---
        app.mv_preview_image = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        print(f"[INFO] Detected {len(detected)} {shape}.")
        app._render_image_on_canvas()

    except Exception as e:
        print(f"[ERROR] detect_shape_generic failed ({shape_type}): {e}")
