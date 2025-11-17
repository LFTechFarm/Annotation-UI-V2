"""
MV/Shapes.py — Unified shape detection module
---------------------------------------------

Supports:
- Hough Circles
- Rectangles (Squares)
- Triangles
- Polygons (N-sided)
- Generic Contours

Each detection updates:
    app.detected_<shape>
    app.mv_preview_image

and triggers app._render_image_on_canvas() for live display.
"""

import cv2
import numpy as np
from PIL import Image


# ============================================================
# === Generic Unified Detection Core =========================
# ============================================================

def detect_shape_generic(app, pil_image, shape_type: str):
    """
    Unified shape detection entrypoint.
    shape_type ∈ {"circles", "rects", "triangles", "polygons", "contours"}
    """
    try:
        shape = shape_type.lower()
        print(f"[DEBUG] Running detect_shape_generic({shape})")

        # --- Convert PIL to OpenCV ---
        img = np.array(pil_image.convert("RGB"))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        vis = img.copy()

        # --- Shared params (safe defaults) ---
        low = int(getattr(app, "rect_canny_low", None).get()) if hasattr(app, "rect_canny_low") else 50
        high = int(getattr(app, "rect_canny_high", None).get()) if hasattr(app, "rect_canny_high") else 150
        min_area = int(getattr(app, "rect_min_area", None).get()) if hasattr(app, "rect_min_area") else 100
        eps = float(getattr(app, "rect_approx_eps", None).get()) if hasattr(app, "rect_approx_eps") else 0.02
        n_sides = int(getattr(app, "poly_n", None).get()) if hasattr(app, "poly_n") else 5

        # ------------------------------------------------------------
        # === Circle Detection (Hough Transform)
        # ------------------------------------------------------------
        if shape == "circles":
            param1 = int(getattr(app, "hough_param1", None).get()) if hasattr(app, "hough_param1") else 100
            param2 = int(getattr(app, "hough_param2", None).get()) if hasattr(app, "hough_param2") else 30
            min_r = int(getattr(app, "hough_min_r", None).get()) if hasattr(app, "hough_min_r") else 10
            max_r = int(getattr(app, "hough_max_r", None).get()) if hasattr(app, "hough_max_r") else 80
            minDist = int(getattr(app, "hough_minDist", None).get()) if hasattr(app, "hough_minDist") else 20

            circles = cv2.HoughCircles(
                blur, cv2.HOUGH_GRADIENT, dp=1, minDist=minDist,
                param1=param1, param2=param2, minRadius=min_r, maxRadius=max_r
            )

            detected = []
            if circles is not None:
                circles = np.uint16(np.around(circles[0, :]))
                for x, y, r in circles:
                    detected.append((int(x), int(y), int(r)))
                    cv2.circle(vis, (x, y), r, (0, 0, 255), 2)

            app.detected_circles = detected
            print(f"[INFO] Detected {len(detected)} circles.")

        # ------------------------------------------------------------
        # === Shapes from contours (rects, triangles, polygons, contours)
        # ------------------------------------------------------------
        else:
            edges = cv2.Canny(blur, low, high)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            detected = []

            for c in contours:
                area = cv2.contourArea(c)
                if area < min_area:
                    continue

                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, eps * peri, True)
                pts = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]
                n = len(approx)

                # --- Shape classification ---
                if shape == "rects" and n == 4:
                    detected.append(pts)
                    cv2.drawContours(vis, [approx], -1, (0, 255, 0), 2)
                elif shape == "triangles" and n == 3:
                    detected.append(pts)
                    cv2.drawContours(vis, [approx], -1, (255, 255, 0), 2)
                elif shape == "polygons" and n == n_sides:
                    detected.append(pts)
                    cv2.drawContours(vis, [approx], -1, (255, 128, 0), 2)
                elif shape == "contours":
                    detected.append(pts)
                    cv2.drawContours(vis, [approx], -1, (0, 255, 255), 1)

            # --- Assign results to correct app attribute ---
            if shape == "rects":
                app.detected_rects = detected
            elif shape == "triangles":
                app.detected_triangles = detected
            elif shape == "polygons":
                app.detected_polygons = detected
            elif shape == "contours":
                app.detected_contours = detected

            print(f"[INFO] Detected {len(detected)} {shape}.")

        # --- Update live preview image ---
        app.mv_preview_image = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        app._render_image_on_canvas()

    except Exception as e:
        print(f"[ERROR] detect_shape_generic failed ({shape_type}): {e}")


# ============================================================
# === Backward-compatible Wrappers ===========================
# ============================================================

def circle_detect(app, img):
    detect_shape_generic(app, img, "circles")

def rectangle_detect(app, img):
    detect_shape_generic(app, img, "rects")

def triangle_detect(app, img):
    detect_shape_generic(app, img, "triangles")

def polygon_detect(app, img):
    detect_shape_generic(app, img, "polygons")

def contour_detect(app, img):
    detect_shape_generic(app, img, "contours")
