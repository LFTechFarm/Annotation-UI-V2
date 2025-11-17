#!/usr/bin/env python3
# yolo_tkinter_labeler_adaptive.py
#
#
# Dependencies: Pillow (pip install Pillow)
# Optional: ultralytics for YOLO predictions

import os
import glob
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Scale, font
from PIL import Image, ImageTk, ImageDraw, ImageStat, ImageFilter
import colorsys
import yaml
import operator
import cv2, numpy as np
# core imports
from core.utils import yolo_to_pixels, pixels_to_yolo, calculate_iou
from core.models import BBox
from AI.yolo import load_yolo_model, predict_current_image
from MV.eXg import excessive_green_apply, _compute_excessive_green_mask
from MV.Shapes import (
    circle_detect,             # Hough-based
    circle_detect_circularity, # new contour-based
    rectangle_detect,
    triangle_detect,
    polygon_detect,
    contour_detect_generic,
    contour_detect_mask,
)

# ---------- Modes ----------
MODES = {
    "none": {"label": "None", "color": "#444444"},
    "draw": {"label": "Draw", "color": "#168fd5"},
    "move": {"label": "Move/Resize", "color": "#FFA500"},
    "delete": {"label": "Delete", "color": "#FF5555"},
    "validate": {"label": "Validate", "color": "#28a745"},
    "class change": {"label": "Change Class", "color": "#49c8cc"}

}

SEGMENTATION_ONLY_MODES = {
    "modify shape": {"label": "Modify Shape", "color": "#b266ff"}
}
# "zoom": {"label": "Zoom", "color": "#AAAAFF"}
# ---------- Main application ----------
class YoloEditorApp(tk.Tk):
    HANDLE_SIZE = 6
# ---------- MAIN App ----------
    def __init__(self):
        super().__init__()
        self.title("YOLO Annotation Editor - Adaptive")
        self.geometry("1200x800")

        # state
        self.image_folder = None
        self.label_folder = None
        self.prediction_folder = None
        self.image_files = []
        self.current_index = -1
        self.current_image = None
        self.tk_image = None

        # scaling / zoom
        self.display_scale = 1.0
        self.zoom_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        # when True, keep offset_x/offset_y as set by zoom operations instead of recentering
        self._preserve_offset = False
        self.img_w = 0
        self.img_h = 0

        # annotation data
        self.bboxes_gt = {}     # img_path -> [BBox,...]
        self.bboxes_pred = {}   # img_path -> [BBox,...]
        self.bboxes_extra = {}  # img_path -> [BBox,...]  (if you add extras manually)
        self.selected_bbox = None

        # segmentation masks data (per image path -> list of masks)
        # mask: {"cls": int, "points": [(x,y), ...]} in image coordinates
        self.seg_masks_gt = {}
        self.seg_masks_pred = {}
        self.seg_masks_extra = {}

        # dragging state
        self.drag_data = {"mode": None, "start": (0, 0), "bbox": None, "handle": None}
        self._drawing_mode = False
        self._new_rect_id = None
        self._new_rect_start = (0, 0)

        # handle mapping
        self.handle_id_to_info = {}

        # yolo model placeholder
        self.yolo_model = None

        # UI vars
        self.current_mode = tk.StringVar(value="none")
        self.mode_buttons = {}

        # active tab tracking and segmentation drawing state
        self._active_left_tab = tk.StringVar(value="bbox")
        self._seg_is_drawing = False
        self._seg_points = []             # image-space points
        self._seg_preview_poly_id = None  # canvas item id for preview polygon
        self._seg_preview_line_id = None  # canvas item id for current edge
        self.seg_cls_var = tk.StringVar(value="0")
        self.selected_seg_mask = None
        self.selected_seg_mask_image = None

        self.show_mv_preview = tk.BooleanVar(value=True)

        self.show_gt = tk.BooleanVar(value=True)
        self.show_pred = tk.BooleanVar(value=False)  # default unchecked
        self.show_extra = tk.BooleanVar(value=False)

        self.transparency_gt = tk.DoubleVar(value=0.2)
        self.transparency_pred = tk.DoubleVar(value=0.2)
        self.transparency_extra = tk.DoubleVar(value=0.2)

        # segmentation visibility/transparency
        self.seg_show_gt = tk.BooleanVar(value=True)
        self.seg_show_pred = tk.BooleanVar(value=False)
        self.seg_show_extra = tk.BooleanVar(value=False)
        self.seg_transparency_gt = tk.DoubleVar(value=0.3)
        self.seg_transparency_pred = tk.DoubleVar(value=0.3)
        self.seg_transparency_extra = tk.DoubleVar(value=0.3)

        self.check_keep_ratio = tk.BooleanVar(value=True)

        # Adaptive tool selection (legacy, removed)

        self.show_ex_index = tk.BooleanVar(value=False)
        self.show_ex_mask = tk.BooleanVar(value=False)
        self.show_ex_index_mult = tk.BooleanVar(value=False)

        self.transparency_ex_index = tk.DoubleVar(value=0.5)
        self.transparency_ex_mask = tk.DoubleVar(value=0.5)
        self.transparency_ex_index_mult = tk.DoubleVar(value=0.5)
        self.transparency_mv_preview = tk.DoubleVar(value=0.5)

        # Global MV index selection and threshold (for always-on controls)
        idx_names_init = list(self.INDEX_FORMULAS.keys()) if hasattr(self, "INDEX_FORMULAS") else ["ExG"]
        if not idx_names_init:
            idx_names_init = ["ExG"]
        self.mv_index_name = tk.StringVar(value=idx_names_init[0])
        self.mv_idx_thresh = tk.IntVar(value=50)

        # Machine Vision: allow using generated index/mask as detection input
        self.mv_use_index_input = tk.BooleanVar(value=False)
        self.mv_index_source = tk.StringVar(value="mask")  # 'mask' or 'index'

        # Custom RGB index coefficients (for linear combination of channels)
        self.custom_r_coef = tk.DoubleVar(value=1.0)
        self.custom_g_coef = tk.DoubleVar(value=0.0)
        self.custom_b_coef = tk.DoubleVar(value=0.0)
        # Segmentation index mode selector (for now only Custom RGB)
        self.seg_index_mode = tk.StringVar(value="Custom RGB")
        # Mask export settings
        self.mask_save_folder = None
        self.mask_save_folder_var = tk.StringVar(value="")


        self.INDEX_FORMULAS = {
            # Indices de base (excès et différences)
            "ExG": lambda r, g, b: 2*g - r - b,
            "ExR": lambda r, g, b: 1.4*r - g,
            "ExB": lambda r, g, b: 1.4*b - g,
            "ExGR": lambda r, g, b: g - r,
            "ExRB": lambda r, g, b: r - b,
            "ExGB": lambda r, g, b: g - b,

            # Indices normalisés (ratios et différences normalisées)
            "NDI_GR": lambda r, g, b: (g - r)/(g + r + 1e-5),
            "NDI_BR": lambda r, g, b: (b - r)/(g + r + 1e-5),
            "NDI_GB": lambda r, g, b: (g - b)/(g + b + 1e-5),
            "NRB": lambda r, g, b: (r - b)/(r + b + 1e-5),
            "NRGDI": lambda r, g, b: (r - g)/(r + g + 1e-5),
            "NRBDI": lambda r, g, b: (r - b)/(r + g + b + 1e-5),
            "NGRDI_mod": lambda r, g, b: (g - r)/(g + r + b + 1e-5),
            "NGBDI_mod": lambda r, g, b: (g - b)/(g + b + r + 1e-5),

            # Ratios simples
            "GR_Ratio": lambda r, g, b: g/(r+1e-5),
            "GB_Ratio": lambda r, g, b: g/(b+1e-5),
            "RG_Ratio": lambda r, g, b: r/(g+1e-5),
            "RB_Ratio": lambda r, g, b: r/(b+1e-5),
            "BG_Ratio": lambda r, g, b: b/(g+1e-5),
            "BR_Ratio": lambda r, g, b: b/(r+1e-5),

            # Fractions de canal
            "G_Fraction": lambda r, g, b: g/(r+g+b+1e-5),
            "R_Fraction": lambda r, g, b: r/(r+g+b+1e-5),
            "B_Fraction": lambda r, g, b: b/(r+g+b+1e-5),

            # Indices normalisés supplémentaires
            "NGI": lambda r, g, b: g/(r+g+b+1e-5),
            "NRGI": lambda r, g, b: (r + g)/(r + g + b + 1e-5),
            "NRBI": lambda r, g, b: (r + b)/(r + g + b + 1e-5),
            "NEGI": lambda r, g, b: (g - r - b)/(g + r + b + 1e-5),
            "NERI": lambda r, g, b: (r - g - b)/(r + g + b + 1e-5),
            "NEBI": lambda r, g, b: (b - g - r)/(b + g + r + 1e-5),

            # Indices modifiés pour la végétation
            "MGRVI": lambda r, g, b: (g - r)/(g + r + 2*b + 1e-5),
            "MGBVI": lambda r, g, b: (g - b)/(g + b + 2*r + 1e-5),
            "MRGVI": lambda r, g, b: (r - g)/(r + g + 2*b + 1e-5),
        }
        self.shape_dispatch = {
            "circles": circle_detect,                  # Hough-based
            "circularity": circle_detect_circularity,  # NEW: contour-based circular blobs
            "square detection": rectangle_detect,
            "triangle detection": triangle_detect,
            "polygon detection": polygon_detect,
            "contour (mask)": contour_detect_mask,
            "contour (generic)": contour_detect_generic,
        }

        # -------------------------------
        # ?? Persistent detection storage
        # Keeps detected shapes across UI refreshes
        # -------------------------------
        self.mv_detections = {
            "circles": [],
            "rects": [],
            "triangles": [],
            "polygons": [],
            "contours_mask": [],
            "contours_generic": []
            }

        tk._default_root = self




        # build UI
        self._build_ui()

    def _build_ui(self):
        """Build the full user interface — cleaned, working version with fixed slider and nav label."""
        import tkinter as tk
        from tkinter import ttk, font

        bold_font = font.Font(family="TkDefaultFont", size=10, weight="bold")
        self.columnconfigure(1, weight=1)

        # ---- LEFT PANEL (refactored) ----
        self._build_left_panel()
        # ---------- MAIN VIEW / TOOL PANEL ----------
        right = ttk.Frame(self)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # --- Navigation bar (top) ---
        nav_frame = tk.Frame(right, bg="#222")
        nav_frame.pack(fill=tk.X, pady=(6, 2))
        nav_frame.columnconfigure(1, weight=1)

        # --- Top row: navigation buttons + label ---
        nav_top = tk.Frame(nav_frame, bg="#222")
        nav_top.grid(row=0, column=0, columnspan=3, sticky="ew")

        tk.Button(
            nav_top, text="← Prev", command=self.prev_image,
            bg="#444", fg="white", font=("Segoe UI", 10, "bold")
        ).pack(side=tk.LEFT, padx=4, pady=2)

        # ? Center label (shows "1/50 – image.jpg")
        self.lbl_nav_image_name = tk.Label(
            nav_top, text="No image", fg="white", bg="#222",
            font=("Segoe UI", 10, "bold"), anchor="center"
        )
        self.lbl_nav_image_name.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)

        tk.Button(
            nav_top, text="Next →", command=self.next_image,
            bg="#444", fg="white", font=("Segoe UI", 10, "bold")
        ).pack(side=tk.RIGHT, padx=4, pady=2)

        # --- Single slider ---
        self.slider_var = tk.DoubleVar(value=0)
        self.slider = tk.Scale(
            nav_frame, from_=0, to=0, orient="horizontal",
            variable=self.slider_var, command=self._on_slider_move,
            showvalue=False, length=360,
            bg="#222", fg="white", troughcolor="#333",
            highlightthickness=0
        )
        self.slider.grid(row=1, column=0, columnspan=3, sticky="ew", padx=4, pady=(0, 2))

        # --- Image area ---
        image_area = ttk.Frame(right)
        image_area.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.canvas = tk.Canvas(image_area, bg="#222222", cursor="cross")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        # Build the right-side tool panel (wrapper for current implementation)
        self.build_right_tool_panel(image_area)

        # --- Canvas bindings ---
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Configure>", lambda e: self.fit_to_window())

        # Mouse wheel zoom
        self.canvas.bind("<MouseWheel>", self._on_mousewheel_zoom)      # Windows / macOS
        self.canvas.bind("<Button-4>", self._on_mousewheel_zoom)        # Linux scroll up
        self.canvas.bind("<Button-5>", self._on_mousewheel_zoom)        # Linux scroll down

        # Middle mouse drag for panning
        self.canvas.bind("<ButtonPress-2>", self._on_pan_start)
        self.canvas.bind("<B2-Motion>", self._on_pan_move)

        # --- Initial visuals ---
        self.update_mode_buttons_look()

# ---------- dataset stats and legend ----------
    def update_dataset_info_labels(self):
        """Update dataset info labels with precomputed stats."""
        total = getattr(self, "total_classes", 0)
        max_per_image = getattr(self, "max_classes_in_image", 0)
        self.lbl_total_classes.config(text=f"Total classes: {total}")
        self.lbl_max_per_image.config(text=f"Max per image: {max_per_image}")

    def generate_class_colors(self,n_classes=100):
        """
        Generate high-contrast colors while avoiding red & blue tones.
        Returns a dict: {class_id: (r,g,b)}.
        """
        colors = {}
        for i in range(n_classes):
            # Spread hues evenly, skipping red (~0°) and blue (~240°)
            h = (i * 1.618) % 1.0  # golden ratio ensures good spacing
            # Shift hue if in forbidden red/blue range
            if 0.0 <= h <= 0.08 or 0.55 <= h <= 0.72:
                h = (h + 0.15) % 1.0
            s, v = 0.9, 0.95  # strong saturation and brightness
            r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(h, s, v)]
            colors[i] = (r, g, b)
        return colors
   
    def get_dataset_stats(self, label_folder):
        """
        Scan all YOLO label files in the given folder and return:
        - total_unique_classes: number of distinct class IDs across the dataset
        - max_classes_in_a_single_file: the maximum count of distinct class IDs in any single label file

        Returns (total_unique_classes, max_classes_in_a_single_file)
        If folder is invalid or no labels, returns (0, 0).
        """
        import os

        if not label_folder or not os.path.isdir(label_folder):
            return 0, 0

        all_classes = set()
        max_per_image = 0

        for fname in os.listdir(label_folder):
            if not fname.lower().endswith(".txt"):
                continue
            fpath = os.path.join(label_folder, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    classes_in_file = set()
                    for raw in f:
                        line = raw.strip()
                        if not line:
                            continue
                        parts = line.split()
                        # first token should be class id; skip malformed lines
                        if len(parts) >= 1:
                            # allow negative? usually not — only accept ints
                            try:
                                cls_id = int(parts[0])
                            except ValueError:
                                # ignore malformed class id
                                continue
                            classes_in_file.add(cls_id)
                            all_classes.add(cls_id)
                    if len(classes_in_file) > max_per_image:
                        max_per_image = len(classes_in_file)
            except Exception as e:
                # avoid crashing on unreadable files; optionally log or print
                print(f"Warning: failed to read '{fpath}': {e}")
                continue

        total_unique = len(all_classes)
        return total_unique, max_per_image
    
    def update_class_selector(self):
        """Update class combobox while preserving the selected class index."""
        if not hasattr(self, "class_selector"):
            return
        num_classes = int(getattr(self, "total_classes", 0) or 0)
        # Build class options
        class_options = []
        for i in range(num_classes):
            if hasattr(self, "class_names") and i < len(self.class_names) and self.class_names[i]:
                class_options.append(f"{i}: {self.class_names[i]}")
            else:
                class_options.append(f"{i}: class_{i}")
        # Always provide a "new class" slot (at index == num_classes)
        new_class_option = f"{num_classes}: (New class)"
        class_options.append(new_class_option)
        # Determine previously selected index (robustly)
        prev_idx = None
        try:
            prev_idx = int(self.parse_selected_class())
        except Exception:
            prev_idx = None
        # Update combobox values/state
        try:
            self.class_selector["values"] = class_options
            self.class_selector.configure(state="readonly")
        except Exception:
            pass
        # Choose target selection: keep previous if still valid; else clamp
        if isinstance(prev_idx, int) and prev_idx >= 0:
            if num_classes > 0:
                clamped = min(prev_idx, num_classes - 1)
                target = class_options[clamped]
            else:
                target = new_class_option
        else:
            target = class_options[0] if class_options else "0"
        try:
            self.class_selector.set(target)
        except Exception:
            pass

    def update_seg_class_selector(self):
        """Update segmentation class combobox and preserve selected class index."""
        try:
            num_classes = int(getattr(self, "total_classes", 0) or 0)
            class_options = []
            for i in range(num_classes):
                if hasattr(self, "class_names") and i < len(self.class_names) and self.class_names[i]:
                    class_options.append(f"{i}: {self.class_names[i]}")
                else:
                    class_options.append(f"{i}: class_{i}")
            new_class_option = f"{num_classes}: (New class)"
            class_options.append(new_class_option)
            # Determine previous selection (numeric)
            prev_idx = None
            try:
                prev_idx = int(self.parse_selected_seg_class())
            except Exception:
                prev_idx = None
            if hasattr(self, 'seg_class_selector'):
                try:
                    self.seg_class_selector["values"] = class_options
                    self.seg_class_selector.configure(state="readonly")
                except Exception:
                    pass
                # Choose preserved or default selection
                if isinstance(prev_idx, int) and prev_idx >= 0:
                    if num_classes > 0:
                        clamped = min(prev_idx, num_classes - 1)
                        target = class_options[clamped]
                    else:
                        target = new_class_option
                else:
                    target = class_options[0] if class_options else "0"
                try:
                    self.seg_class_selector.set(target)
                except Exception:
                    pass
            # Normalize seg_cls_var to numeric-only
            try:
                txt = self.seg_cls_var.get()
                if isinstance(txt, str) and ':' in txt:
                    self.seg_cls_var.set(txt.split(':',1)[0].strip())
            except Exception:
                pass
        except Exception:
            pass

    def update_class_legend(self):
        """Build or refresh the class color legend based on dataset stats."""
        # Ensure class_colors exists
        if not hasattr(self, "class_colors") or not isinstance(self.class_colors, dict):
            self.class_colors = {}
        for widget in self.legend_frame.winfo_children():
            widget.destroy()
        num_classes = getattr(self, "total_classes", 0)
        if num_classes == 0:
            tk.Label(self.legend_frame, text="No classes found", fg="gray").pack()
            return
        # Ensure enough colors exist
        if len(self.class_colors) < num_classes:
            self.class_colors = self.generate_class_colors(num_classes)

        # Use 3 columns
        cols = 3
        rows = (num_classes + cols - 1) // cols  # Calculate the number of rows needed

        for i in range(num_classes):
            c = self.class_colors.get(i, (160, 160, 160))
            name = (
                self.class_names[i] if hasattr(self, "class_names") and i < len(self.class_names)
                else f"class_{i}"
            )
            lbl = tk.Label(
                self.legend_frame,
                text=name,
                bg=f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}",
                fg="white",
                width=10,
                relief="groove"
            )
            # Place labels in a 3-column grid
            lbl.grid(row=i // cols, column=i % cols, padx=2, pady=2)

    def update_dataset_info(self):
        """
        Recompute total_classes from existing label files + any created bboxes,
        regenerate colors, update labels, combobox and legend.
        """
        # Base from label files
        label_folder = getattr(self, "label_folder", None)
        total_from_files = 0
        try:
            if label_folder:
                total_from_files, _ = self.get_dataset_stats(label_folder)
        except Exception:
            total_from_files = getattr(self, "total_classes", 0) or 0

        # Examine in-memory bboxes to detect any higher class IDs
        max_cls = -1
        for img, boxes in getattr(self, "bboxes_gt", {}).items():
            for b in boxes:
                cid = getattr(b, "cls", None)
                if cid is None:
                    cid = getattr(b, "class_id", None)
                if cid is not None:
                    try:
                        max_cls = max(max_cls, int(cid))
                    except Exception:
                        pass
        # Include segmentation mask classes (GT) in total
        try:
            for img, masks in getattr(self, "seg_masks_gt", {}).items():
                for mk in (masks or []):
                    cid = mk.get("cls", None)
                    if cid is not None:
                        try:
                            max_cls = max(max_cls, int(cid))
                        except Exception:
                            pass
        except Exception:
            pass
        total = max(total_from_files, max_cls + 1, 0)

        # Ensure class_names includes all classes up to total
        if hasattr(self, "class_names"):
            # Extend class_names with default names for any missing classes
            while len(self.class_names) < total:
                self.class_names.append(f"class_{len(self.class_names)}")
        else:
            self.class_names = [f"class_{i}" for i in range(total)]

        # Store
        self.total_classes = total

        # Update labels
        if hasattr(self, "lbl_total_classes"):
            self.lbl_total_classes.config(text=f"Total classes: {self.total_classes}")

        # Ensure class_colors is sized to total
        if not hasattr(self, "class_colors") or not isinstance(self.class_colors, dict):
            self.class_colors = {}
        if len(self.class_colors) < max(1, self.total_classes):
            self.class_colors = self.generate_class_colors(max(1, self.total_classes))

        # Update comboboxes
        if hasattr(self, "class_selector"):
            self.update_class_selector()
        if hasattr(self, "seg_class_selector"):
            self.update_seg_class_selector()
        if hasattr(self, "seg_class_selector"):
            self.update_seg_class_selector()

        # Update legend
        if hasattr(self, "legend_frame"):
            self.update_class_legend()

    # ---------- Folder & files ----------
    def select_parent_folder(self):
        """Select the dataset folder, scan class stats, and update the interface."""
        folder = filedialog.askdirectory(
            title="Select Parent Folder (must contain 'images' and 'labels' subfolders)"
        )
        if not folder:
            return

        # Expected structure
        self.image_folder = os.path.join(folder, "images")
        self.label_folder = os.path.join(folder, "labels")
        self.prediction_folder = os.path.join(folder, "predictions")
        os.makedirs(self.prediction_folder, exist_ok=True)

        # Validate folder structure
        if not os.path.exists(self.image_folder) or not os.path.exists(self.label_folder):
            messagebox.showerror("Error", "Parent folder must contain 'images' and 'labels' subfolders.")
            return

        # ---- Load dataset statistics ----
        self.total_classes, self.max_classes_in_image = self.get_dataset_stats(self.label_folder)
        print(f"[INFO] Found {self.total_classes} unique classes across labels")

        # ---- Try to load data.yaml (optional) ----
        yaml_path = os.path.join(folder, "data.yaml")
        if os.path.exists(yaml_path):
            try:
                with open(yaml_path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                self.class_names = data.get("names", [])
                if not self.total_classes and "nc" in data:
                    self.total_classes = data["nc"]
                print(f"? Loaded {len(self.class_names)} class names from data.yaml")
            except Exception as e:
                print(f"[WARN] Failed to read {yaml_path}: {e}")
                self.class_names = []
        else:
            self.class_names = []

        # ---- Update UI elements ----
        self.update_dataset_info_labels()
        self.update_class_legend()

        # If dropdown exists, refresh it with new classes
        if hasattr(self, "update_class_selector"):
            self.update_class_selector()

        # ---- Load image and annotation data ----
        self.load_image_list()
        self.current_index = 0 if self.image_files else -1
        self.load_annotations_for_all_images()
        # Compute dataset-level stats now that images/labels are known
        try:
            self.refresh_dataset_info()
        except Exception:
            pass
        self.show_image_at_index()
    
    def load_image_list(self):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
        files = []
        for e in exts:
            files.extend(sorted(glob.glob(os.path.join(self.image_folder, e))))
        self.image_files = files
        # reset dicts
        self.bboxes_gt = {}
        self.bboxes_pred = {}
        self.bboxes_extra = {}

    def load_annotations_for_all_images(self):
        if not self.label_folder:
            return
        for img_path in self.image_files:
            fname = os.path.splitext(os.path.basename(img_path))[0]
            txt_path = os.path.join(self.label_folder, fname + ".txt")
            bboxes = []
            if os.path.exists(txt_path):
                try:
                    with open(txt_path, "r") as f:
                        lines = [l.strip() for l in f.readlines() if l.strip()]
                    with Image.open(img_path) as im:
                        w, h = im.size
                    for L in lines:
                        parts = L.split()
                        if len(parts) >= 5:
                            cls = int(float(parts[0]))
                            nx = float(parts[1])
                            ny = float(parts[2])
                            nw = float(parts[3])
                            nh = float(parts[4])
                            x1, y1, x2, y2 = yolo_to_pixels(nx, ny, nw, nh, w, h)
                            bboxes.append(BBox(cls, x1, y1, x2, y2))
                except Exception as e:
                    print("Error reading", txt_path, e)
            self.bboxes_gt[img_path] = bboxes
            
    def load_current_image(self):
        if not self.image_files or self.current_index < 0:
            self.current_image = None
            return
        img_path = self.image_files[self.current_index]
        try:
            img = Image.open(img_path).convert("RGB")
            self.current_image = img
            self.img_w, self.img_h = img.size
        except Exception as e:
            messagebox.showerror("Error", f"Could not open image:\n{img_path}\n{e}")
            self.current_image = None

    def load_yaml_file(self):
        """Open a file dialog to select a YOLO-style data.yaml file and load it."""
        file_path = filedialog.askopenfilename(
            title="Select data.yaml",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")]
        )
        if not file_path:
            return  # User cancelled
        try:
            with open(file_path, "r") as f:
                data = yaml.safe_load(f)
            yaml_class_names = data.get("names", [])
            yaml_total_classes = data.get("nc", len(yaml_class_names))

            # Merge YAML classes with existing dataset classes
            if hasattr(self, "class_names") and self.class_names:
                # Combine existing class names with YAML class names
                combined_class_names = self.class_names.copy()
                for i, name in enumerate(yaml_class_names):
                    if i < len(combined_class_names):
                        # Update existing class name if YAML has a name for this index
                        if name and combined_class_names[i] != name:
                            combined_class_names[i] = name
                    else:
                        # Add new class names from YAML
                        combined_class_names.append(name)
                self.class_names = combined_class_names
            else:
                self.class_names = yaml_class_names

            # Update total_classes to the maximum of YAML and dataset
            dataset_total_classes = getattr(self, "total_classes", 0)
            self.total_classes = max(yaml_total_classes, dataset_total_classes)

            # Refresh legend and dropdown
            self.update_class_legend()
            self.update_class_selector()
            print(f"? Loaded {len(self.class_names)} class names from {file_path}")
        except Exception as e:
            print(f"? Failed to load YAML file: {e}")

# ---------- Adaptive Tool Panel ----------
    def _build_index_selection(self, parent):
        import tkinter as tk
        from tkinter import ttk

        # ---------- Base frame ----------
        tool_frame = ttk.Frame(parent, width=280, relief=tk.GROOVE, borderwidth=1)
        tool_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(2, 0))
        tool_frame.pack_propagate(False)

        # (Readability controls now live inside the Machine Vision tab)

        # ---------- Notebook tabs ----------
        self.right_tabs = ttk.Notebook(tool_frame)
        self.right_tabs.pack(fill="both", expand=True, padx=4, pady=4)

        self.tab_dataset = ttk.Frame(self.right_tabs)
        self.tab_yolo = ttk.Frame(self.right_tabs)
        self.tab_mv = ttk.Frame(self.right_tabs)

        self.right_tabs.add(self.tab_dataset, text="🗂 Dataset & Image")
        self.right_tabs.add(self.tab_yolo, text="🤖 YOLO Model")
        self.right_tabs.add(self.tab_mv, text="🔬 Machine Vision")

        # ---------- Build each tab ----------
        self._build_dataset_tab(self.tab_dataset)
        self._build_yolo_tab(self.tab_yolo)
        self._build_mv_tab(self.tab_mv)

    # Backward-compatible wrapper used by _build_ui
    def build_right_tool_panel(self, parent):
        try:
            return self._build_index_selection(parent)
        except Exception as e:
            print(f"[WARN] build_right_tool_panel failed: {e}")
            # Ensure UI doesn't break entirely
            pass

    def _build_dataset_tab(self, parent):
        """Build Dataset & Image Info tab (clean + fixed version)."""
        import tkinter as tk
        from tkinter import ttk
        from datetime import datetime
        import os

        ttk.Label(parent, text="🗂 Dataset Information", font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W, pady=(0, 4))
        ttk.Button(parent, text="Load data.yaml", command=self.load_yaml_file).pack(fill=tk.X, pady=(0, 6))

        # Basic dataset info
        self.lbl_total_classes = ttk.Label(parent, text="Total classes: -", anchor=tk.W)
        self.lbl_total_classes.pack(fill=tk.X)
        self.lbl_max_per_image = ttk.Label(parent, text="Max per image: -", anchor=tk.W)
        self.lbl_max_per_image.pack(fill=tk.X)
        self.lbl_total_images = ttk.Label(parent, text="Total images: -", anchor=tk.W)
        self.lbl_total_images.pack(fill=tk.X)

        # Dataset-level statistics
        ttk.Separator(parent).pack(fill=tk.X, pady=6)
        ttk.Label(parent, text="📈 Dataset Statistics", font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W, pady=(2, 4))
        self.lbl_avg_boxes = ttk.Label(parent, text="Avg boxes per image: -", anchor=tk.W)
        self.lbl_avg_boxes.pack(fill=tk.X)
        self.lbl_total_annotations = ttk.Label(parent, text="Total annotations: -", anchor=tk.W)
        self.lbl_total_annotations.pack(fill=tk.X)
        self.lbl_class_balance = ttk.Label(parent, text="Class balance: -", anchor=tk.W)
        self.lbl_class_balance.pack(fill=tk.X)

        # Current image info
        ttk.Separator(parent).pack(fill=tk.X, pady=6)
        ttk.Label(parent, text="🖼️ Current Image", font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W)
        self.lbl_image_name = ttk.Label(parent, text="No image loaded", anchor=tk.W)
        self.lbl_image_name.pack(fill=tk.X, pady=(0, 4))
        self.lbl_counts_info = ttk.Label(parent, text="GT: 0   Pred: 0   Extra: 0", anchor=tk.W)
        self.lbl_counts_info.pack(fill=tk.X)

        # Image stats — only GT BBox coverage
        ttk.Separator(parent).pack(fill=tk.X, pady=6)
        ttk.Label(parent, text="📊 Image Statistics", font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W)
        self.lbl_bbox_coverage = ttk.Label(parent, text="GT Coverage: -", anchor=tk.W)
        self.lbl_bbox_coverage.pack(fill=tk.X)
        self.lbl_capture_date = ttk.Label(parent, text=f"Date: {datetime.now().strftime('%Y-%m-%d')}", anchor=tk.W)
        self.lbl_capture_date.pack(fill=tk.X)

        # Class color legend
        ttk.Separator(parent).pack(fill=tk.X, pady=(6, 4))
        ttk.Label(parent, text="🎨 Class Legend", font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W, pady=(0, 4))
        self.legend_frame = tk.Frame(parent)
        self.legend_frame.pack(fill=tk.X, pady=(4, 2))

        # --- Live update logic ---
        def _update_dataset_stats():
            """Refresh dataset and current image info (fixed)."""
            # Dataset stats
            info = getattr(self, "dataset_info", {})
            total_images = len(getattr(self, "image_files", []))
            total_ann = info.get("total_annotations", 0)
            avg_boxes = info.get("avg_boxes", "-")
            class_balance = info.get("class_balance", "-")

            self.lbl_total_images.config(text=f"Total images: {total_images}")
            self.lbl_total_annotations.config(text=f"Total annotations: {total_ann}")
            self.lbl_avg_boxes.config(text=f"Avg boxes per image: {avg_boxes}")
            self.lbl_class_balance.config(text=f"Class balance: {class_balance}")

            # Current image coverage and counts
            try:
                img_path = self.image_files[self.current_index] if getattr(self, "image_files", None) and self.current_index >= 0 else None
                gt_list = self.bboxes_gt.get(img_path, []) if img_path else []
                pred_list = self.bboxes_pred.get(img_path, []) if img_path else []
                extra_list = self.bboxes_extra.get(img_path, []) if img_path else []
                if hasattr(self, "current_image") and self.current_image is not None:
                    W, H = self.current_image.size
                    total_area = max(1, W * H)
                    bbox_area = 0
                    for bb in gt_list:
                        try:
                            bbox_area += max(0, (bb.x2 - bb.x1)) * max(0, (bb.y2 - bb.y1))
                        except Exception:
                            pass
                    coverage = 100.0 * bbox_area / total_area
                    self.lbl_bbox_coverage.config(text=f"GT Coverage: {coverage:.2f}%")
                else:
                    self.lbl_bbox_coverage.config(text="GT Coverage: -")

                self.lbl_counts_info.config(text=f"GT: {len(gt_list)}   Pred: {len(pred_list)}   Extra: {len(extra_list)}")

                if img_path:
                    self.lbl_image_name.config(text=os.path.basename(img_path))
                else:
                    self.lbl_image_name.config(text="No image loaded")
            except Exception:
                pass

        # Expose updater
        self.update_dataset_stats = _update_dataset_stats

    def _build_yolo_tab(self, parent):
        """Build YOLO Model tab."""
        from tkinter import ttk
        ttk.Label(parent, text="YOLO Detection", font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W, pady=(0, 6))

        ttk.Button(parent, text="Load YOLO Model", command=lambda: load_yolo_model(self)).pack(fill=tk.X, pady=2)
        ttk.Button(parent, text="Predict Current Image", command=lambda: predict_current_image(self)).pack(fill=tk.X, pady=2)
        ttk.Button(parent, text="Validate Selected Prediction", command=self.validate_selected_prediction).pack(fill=tk.X, pady=2)
        ttk.Button(parent, text="Validate All Predictions", command=self.validate_all_predictions).pack(fill=tk.X, pady=2)
    
    # --- MV tab ---
    def _build_mv_tab(self, parent):
        """
        Build Machine Vision tab subdivided into three pages:
        1?? Index Computation
        2?? Segmentation
        3?? Shape Detection
        """
        import tkinter as tk
        from tkinter import ttk

        # --- Notebook for subpages ---
        mv_notebook = ttk.Notebook(parent)
        mv_notebook.pack(fill="both", expand=True, padx=4, pady=4)

        tab_index = ttk.Frame(mv_notebook)
        tab_segment = ttk.Frame(mv_notebook)
        tab_shape = ttk.Frame(mv_notebook)

        mv_notebook.add(tab_index, text="1. Index")
        mv_notebook.add(tab_segment, text="2. Mask")
        mv_notebook.add(tab_shape, text="3. Shape")

        # --- Delegate to dedicated functions ---
        self._build_mv_index_page(tab_index)
        self._build_mv_segmentation_page(tab_segment)
        self._build_mv_shape_page(tab_shape)
    
    def _build_mv_segmentation_page(self, parent):
        """Build Step 2 – Segmentation page."""
        import tkinter as tk
        from tkinter import ttk

        ttk.Label(parent, text="Step 2️⃣ - Mask", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(2, 4))

        # Readability section for Segmentation tab
        try:
            self._build_mv_readability(parent)
        except Exception:
            pass
        self.show_ex_mask = getattr(self, "show_ex_mask", tk.BooleanVar(value=True))
        self.transparency_ex_mask = getattr(self, "transparency_ex_mask", tk.DoubleVar(value=0.5))

        # --- Mask export controls ---
        export = ttk.LabelFrame(parent, text="Mask Export")
        export.pack(fill="x", pady=(6, 4))

        # Row: choose folder
        row = ttk.Frame(export); row.pack(fill="x", pady=(2,2))
        ttk.Label(row, text="Output folder:").pack(side="left")
        entry = ttk.Entry(row, textvariable=self.mask_save_folder_var, state="readonly")
        entry.pack(side="left", expand=True, fill="x", padx=(6,6))
        def _choose_mask_folder():
            from tkinter import filedialog
            folder = filedialog.askdirectory(title="Select folder to save masks")
            if folder:
                self.mask_save_folder = folder
                try:
                    self.mask_save_folder_var.set(folder)
                except Exception:
                    pass
        ttk.Button(row, text="Choose Folder", command=_choose_mask_folder).pack(side="left")

        # Row: save current mask
        btn_row = ttk.Frame(export); btn_row.pack(fill="x", pady=(2,2))
        ttk.Button(btn_row, text="Save Current Mask", command=self.save_current_mask).pack(side="left")

    def _build_mv_index_page(self, parent):
        """Build Step 1 – Index Computation page."""
        import tkinter as tk
        from tkinter import ttk

        ttk.Label(parent, text="Step 1️⃣ - Index", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(2, 4))
        try:
            self._build_mv_readability(parent)
        except Exception:
            pass

    def _build_mv_shape_page(self, parent):
        """Build Step 3️⃣ - Shape Detection page (with input source picker)."""
        import tkinter as tk
        from tkinter import ttk

        ttk.Label(parent, text="Step 3️⃣ - Shape", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", pady=(2, 4))
        try:
            self._build_mv_readability(parent)
        except Exception:
            pass
        # --- Detection input (band/source) ---
        row_src = ttk.Frame(parent)
        row_src.pack(fill="x", pady=(0, 6))
        ttk.Label(row_src, text="Detection input:").pack(side="left")

        # Persisted selection
        self.mv_input_source = getattr(self, "mv_input_source", tk.StringVar(value="RGB"))
        src_combo = ttk.Combobox(
            row_src, textvariable=self.mv_input_source,
            values=["RGB", "Index", "Mask"],
            state="readonly", width=12
        )
        src_combo.pack(side="left", padx=6)

        # --- Handle change of source ---
        def _on_mv_input_changed(*_):
            val = (self.mv_input_source.get() or "RGB").lower()
            self.mv_use_index_input = getattr(self, "mv_use_index_input", tk.BooleanVar(value=False))
            self.mv_use_index_input.set(val in ("index", "mask"))

            self.mv_index_source = getattr(self, "mv_index_source", tk.StringVar(value="rgb"))
            self.mv_index_source.set(val if val in ("index", "mask") else "rgb")

            self._render_image_on_canvas()

        self.mv_input_source.trace_add("write", _on_mv_input_changed)

        # --- Shape selection dropdown ---
        # ? Ensure circularity is always present before UI builds
        if not hasattr(self, "shape_dispatch"):
            self.shape_dispatch = {}
        if "circularity detection" not in self.shape_dispatch:
            try:
                from MV.Shapes import circle_detect_circularity
                self.shape_dispatch["circularity detection"] = circle_detect_circularity
            except Exception as e:
                print(f"[WARN] Could not register circularity detection: {e}")

        shapes = list(self.shape_dispatch.keys())
        self.mv_shape_name = getattr(self, "mv_shape_name", tk.StringVar(value=shapes[0]))

        frm = ttk.Frame(parent)
        frm.pack(fill="x", pady=(2, 4))
        ttk.Label(frm, text="Shape:").pack(side="left")
        combo = ttk.Combobox(frm, textvariable=self.mv_shape_name, values=shapes, state="readonly", width=20)
        combo.pack(side="left", padx=6)

        # --- Dynamic parameter frame ---
        self.mv_param_frame = ttk.Frame(parent)
        self.mv_param_frame.pack(fill="x", pady=(4, 6))

        def rebuild_shape_params(event=None):
            for child in self.mv_param_frame.winfo_children():
                child.destroy()

            shape = (self.mv_shape_name.get() or "").lower()

            # --- Circles ---
            if "circles" in shape:
                ttk.Label(self.mv_param_frame, text="Hough Circle Parameters", font=("TkDefaultFont", 9, "bold")).pack(anchor="w", pady=(0, 2))
                self.hough_param1  = getattr(self, "hough_param1",  tk.IntVar(value=100))
                self.hough_param2  = getattr(self, "hough_param2",  tk.IntVar(value=30))
                self.hough_min_r   = getattr(self, "hough_min_r",   tk.IntVar(value=10))
                self.hough_max_r   = getattr(self, "hough_max_r",   tk.IntVar(value=80))
                self.hough_minDist = getattr(self, "hough_minDist", tk.IntVar(value=20))

                self.make_param_slider("Canny Threshold (param1)", self.hough_param1, 10, 300,
                                    callback=lambda: self._mv_run_preview('circles'))
                self.make_param_slider("Accumulator Threshold (param2)", self.hough_param2, 5, 100,
                                    callback=lambda: self._mv_run_preview('circles'))
                self.make_param_slider("Min Radius", self.hough_min_r, 0, 100,
                                    callback=lambda: self._mv_run_preview('circles'))
                self.make_param_slider("Max Radius", self.hough_max_r, 0, 200,
                                    callback=lambda: self._mv_run_preview('circles'))
                self.make_param_slider("Min Distance", self.hough_minDist, 1, 200,
                                    callback=lambda: self._mv_run_preview('circles'))

            # --- Circularity ---
            elif "circularity" in shape:
                ttk.Label(self.mv_param_frame, text="Circularity Detection Parameters", font=("TkDefaultFont", 9, "bold")).pack(anchor="w", pady=(0, 2))
                self.circularity_threshold = getattr(self, "circularity_threshold", tk.DoubleVar(value=0.7))
                self.circularity_min_area  = getattr(self, "circularity_min_area",  tk.IntVar(value=30))
                self.hough_min_r = getattr(self, "hough_min_r", tk.IntVar(value=5))
                self.hough_max_r = getattr(self, "hough_max_r", tk.IntVar(value=80))
                # --- Circularity Threshold (float slider with live label)
                row_thr = ttk.Frame(self.mv_param_frame)
                row_thr.pack(fill="x", pady=(2, 2))
                ttk.Label(row_thr, text="Circularity Threshold").pack(side="left")
                thr_val_label = ttk.Label(row_thr, text=f"{self.circularity_threshold.get():.2f}")
                thr_val_label.pack(side="right")
                thr_slider = ttk.Scale(row_thr, from_=0.1, to=1.0, orient="horizontal",
                                    variable=self.circularity_threshold, length=140)
                thr_slider.pack(side="left", padx=6, fill="x", expand=True)
                self.circularity_threshold.trace_add(
                    "write",
                    lambda *_: (
                        thr_val_label.config(text=f"{self.circularity_threshold.get():.2f}"),
                        self._mv_run_preview("circularity")
                    )
                )

                # --- Min Contour Area
                row_area = ttk.Frame(self.mv_param_frame)
                row_area.pack(fill="x", pady=(2, 2))
                ttk.Label(row_area, text="Min Contour Area").pack(side="left")
                area_val_label = ttk.Label(row_area, text=f"{self.circularity_min_area.get()}")
                area_val_label.pack(side="right")
                area_slider = ttk.Scale(row_area, from_=10, to=500, orient="horizontal",
                                        variable=self.circularity_min_area, length=140)
                area_slider.pack(side="left", padx=6, fill="x", expand=True)
                self.circularity_min_area.trace_add(
                    "write",
                    lambda *_: (
                        area_val_label.config(text=f"{int(self.circularity_min_area.get())}"),
                        self._mv_run_preview("circularity")
                    )
                )

                # --- Radius sliders (reuse make_param_slider)
                self.make_param_slider("Min Radius", self.hough_min_r, 1, 100,
                                    callback=lambda: self._mv_run_preview("circularity"))
                self.make_param_slider("Max Radius", self.hough_max_r, 1, 200,
                                    callback=lambda: self._mv_run_preview("circularity"))

            # --- Squares ---
            elif "square" in shape:
                ttk.Label(self.mv_param_frame, text="Square Detection Parameters", font=("TkDefaultFont", 9, "bold")).pack(anchor="w", pady=(0, 2))
                self.rect_canny_low  = getattr(self, "rect_canny_low",  tk.IntVar(value=50))
                self.rect_canny_high = getattr(self, "rect_canny_high", tk.IntVar(value=150))
                self.rect_min_area   = getattr(self, "rect_min_area",   tk.IntVar(value=100))
                self.rect_approx_eps = getattr(self, "rect_approx_eps", tk.DoubleVar(value=0.02))

                self.make_param_slider("Canny low", self.rect_canny_low, 0, 200,
                                    callback=lambda: self._mv_run_preview('Square Detection'))
                self.make_param_slider("Canny high", self.rect_canny_high, 0, 300,
                                    callback=lambda: self._mv_run_preview('Square Detection'))
                self.make_param_slider("Min area", self.rect_min_area, 1, 10000,
                                    callback=lambda: self._mv_run_preview('Square Detection'))
                self.make_param_slider("Approx e", self.rect_approx_eps, 0.005, 0.1,
                                    fmt=lambda v: f"{float(v):.4f}",
                                    callback=lambda: self._mv_run_preview('Square Detection'))

            elif "triangle" in shape:
                ttk.Label(self.mv_param_frame, text="Triangle Detection Parameters", font=("TkDefaultFont", 9, "bold")).pack(anchor="w", pady=(0, 2))

            elif "polygon" in shape:
                ttk.Label(self.mv_param_frame, text="Polygon Detection Parameters", font=("TkDefaultFont", 9, "bold")).pack(anchor="w", pady=(0, 2))
                self.poly_n = getattr(self, "poly_n", tk.IntVar(value=5))
                self.make_param_slider("Polygon vertices (N)", self.poly_n, 3, 12,
                                    callback=lambda: self._mv_run_preview('Polygon Detection'))

            # --- Reset detections ---
            if hasattr(self, "clear_mv_preview"):
                self.clear_mv_preview()
            else:
                self.mv_preview_image = None
                for attr in ("detected_circles", "detected_circularity", "detected_rects",
                            "detected_triangles", "detected_polygons"):
                    setattr(self, attr, [])
            self._render_image_on_canvas()

        combo.bind("<<ComboboxSelected>>", rebuild_shape_params)
        rebuild_shape_params()
        ttk.Button(parent, text="Preview Shape Detection",
                command=lambda: self._mv_run_and_maybe_add(self.mv_shape_name.get())
                ).pack(fill="x", pady=(4, 2))
        # --- Post-processing buttons ---
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill="x", pady=(0, 4))


        ttk.Button(btn_frame, text="Get Bounding Boxes",
                command=lambda: self._mv_preview_bboxes(self.mv_shape_name.get())
                ).pack(side="left", expand=True, fill="x", padx=(0, 3))
        ttk.Button(btn_frame, text="Send to Extras",
                command=lambda: self._mv_add_shape_to_extras(self.mv_shape_name.get())
                ).pack(side="left", expand=True, fill="x", padx=(3, 0))

    def _build_mv_readability(self, parent):
        import tkinter as tk
        from tkinter import ttk

        rb = ttk.LabelFrame(parent, text="Readability")
        rb.pack(fill="x", pady=(4, 4))

        # Ensure shared vars exist
        if not hasattr(self, "show_ex_index"):
            self.show_ex_index = tk.BooleanVar(value=True)
        if not hasattr(self, "show_ex_mask"):
            self.show_ex_mask = tk.BooleanVar(value=True)
        if not hasattr(self, "show_ex_index_mult"):
            self.show_ex_index_mult = tk.BooleanVar(value=False)
        if not hasattr(self, "show_mv_preview"):
            self.show_mv_preview = tk.BooleanVar(value=True)
        if not hasattr(self, "transparency_ex_index"):
            self.transparency_ex_index = tk.DoubleVar(value=0.7)
        if not hasattr(self, "transparency_ex_mask"):
            self.transparency_ex_mask = tk.DoubleVar(value=0.5)
        if not hasattr(self, "transparency_ex_index_mult"):
            self.transparency_ex_index_mult = tk.DoubleVar(value=0.5)
        if not hasattr(self, "transparency_mv_preview"):
            self.transparency_mv_preview = tk.DoubleVar(value=0.5)

        # Index selection and generate (now supports 'Custom RGB')
        idx_names = list(getattr(self, "INDEX_FORMULAS", {}).keys()) or ["ExG"]
        if "Custom RGB" not in idx_names:
            idx_names.append("Custom RGB")
        self.mv_index_name = getattr(self, "mv_index_name", tk.StringVar(value=idx_names[0]))
        row_idx = ttk.Frame(rb); row_idx.pack(fill="x", pady=2)
        ttk.Label(row_idx, text="Index:").pack(side="left")
        ttk.Combobox(row_idx, textvariable=self.mv_index_name, values=idx_names, state="readonly", width=16).pack(side="left", padx=(6, 0))
        ttk.Button(row_idx, text="Generate", command=lambda: self.generate_index_image(self.mv_index_name.get())).pack(side="left", padx=(6, 0))

        # Custom RGB coefficient sliders (visible only when 'Custom RGB' is selected)
        custom_frame = ttk.LabelFrame(rb, text="Custom RGB Coefficients")
        # Build rows once; show/hide via trace
        def _build_custom_rows():
            # Clear previous
            for w in custom_frame.winfo_children():
                w.destroy()
            # R
            rrow = ttk.Frame(custom_frame); rrow.pack(fill="x", pady=1)
            ttk.Label(rrow, text="R coef").pack(side="left")
            rval = ttk.Label(rrow, text=f"{self.custom_r_coef.get():.2f}"); rval.pack(side="right")
            ttk.Scale(rrow, from_=-3.0, to=3.0, orient="horizontal", variable=self.custom_r_coef, length=160,
                      command=lambda e: (rval.config(text=f"{self.custom_r_coef.get():.2f}"), self.generate_index_image("Custom RGB"))).pack(side="right", padx=6)
            # G
            grow = ttk.Frame(custom_frame); grow.pack(fill="x", pady=1)
            ttk.Label(grow, text="G coef").pack(side="left")
            gval = ttk.Label(grow, text=f"{self.custom_g_coef.get():.2f}"); gval.pack(side="right")
            ttk.Scale(grow, from_=-3.0, to=3.0, orient="horizontal", variable=self.custom_g_coef, length=160,
                      command=lambda e: (gval.config(text=f"{self.custom_g_coef.get():.2f}"), self.generate_index_image("Custom RGB"))).pack(side="right", padx=6)
            # B
            brow = ttk.Frame(custom_frame); brow.pack(fill="x", pady=1)
            ttk.Label(brow, text="B coef").pack(side="left")
            bval = ttk.Label(brow, text=f"{self.custom_b_coef.get():.2f}"); bval.pack(side="right")
            ttk.Scale(brow, from_=-3.0, to=3.0, orient="horizontal", variable=self.custom_b_coef, length=160,
                      command=lambda e: (bval.config(text=f"{self.custom_b_coef.get():.2f}"), self.generate_index_image("Custom RGB"))).pack(side="right", padx=6)

        def _refresh_custom_visibility(*_):
            name = (self.mv_index_name.get() or "").strip().lower()
            if name == "custom rgb" or name.replace(" ","") == "customrgb":
                if not custom_frame.winfo_ismapped():
                    custom_frame.pack(fill="x", pady=(4,4))
                    _build_custom_rows()
            else:
                if custom_frame.winfo_ismapped():
                    custom_frame.pack_forget()

        try:
            self.mv_index_name.trace_add("write", _refresh_custom_visibility)
        except Exception:
            pass
        _refresh_custom_visibility()

        # Rows helper
        def rb_row(label, var_show, var_alpha):
            r = ttk.Frame(rb); r.pack(fill="x", pady=1)
            ttk.Checkbutton(r, text=label, variable=var_show, command=self._render_image_on_canvas).pack(side="left")
            ttk.Scale(r, from_=0, to=1, orient="horizontal", variable=var_alpha, length=140,
                      command=lambda e: self._render_image_on_canvas()).pack(side="right")

        # Index, Mask, RGB*Mask, MV Preview
        rb_row("Index", self.show_ex_index, self.transparency_ex_index)
        rb_row("Mask", self.show_ex_mask, self.transparency_ex_mask)
        rb_row("RGB*Mask", self.show_ex_index_mult, self.transparency_ex_index_mult)
        rb_row("MV Preview", self.show_mv_preview, self.transparency_mv_preview)

        # Threshold
        if not hasattr(self, "mv_idx_thresh"):
            self.mv_idx_thresh = tk.IntVar(value=50)
        thr_row = ttk.Frame(rb); thr_row.pack(fill="x", pady=(2, 2))
        ttk.Label(thr_row, text="Threshold").pack(side="left")
        thr_val = ttk.Label(thr_row, text=str(int(self.mv_idx_thresh.get())))
        thr_val.pack(side="right")
        ttk.Scale(thr_row, from_=0, to=255, orient="horizontal", variable=self.mv_idx_thresh,
                  command=lambda e: self._apply_threshold_to_index_fast(self.mv_index_name.get()), length=140).pack(fill="x", padx=(6,6))
        try:
            self.mv_idx_thresh.trace_add("write", lambda *a: thr_val.config(text=str(int(self.mv_idx_thresh.get()))))
        except Exception:
            pass

   # ---------- Navigation Label ----------
    def update_image_label(self):
        """Sync the top navigation label with the current image info."""
        if not hasattr(self, "lbl_image_name"):
            return
        if not getattr(self, "image_files", []):
            self.lbl_image_name.config(text="No image")
            return

        img_name = os.path.basename(getattr(self, "current_image_path", ""))
        total = len(self.image_files)
        idx = self.current_index + 1 if self.current_index >= 0 else 0
        self.lbl_image_name.config(text=f"{idx}/{total}  {img_name}")

   # ---------- UI Utilities ----------
    def make_param_slider(self, label, var, min_val, max_val, callback=None, fmt=None, resolution=None):
        """
        Create a labeled slider (float or int) with optional callback.
        Supports both integer and float variables.
        """
        import tkinter as tk
        from tkinter import ttk

        frame = ttk.Frame(self.mv_param_frame)
        frame.pack(fill="x", pady=(2, 2))
        ttk.Label(frame, text=label).pack(side="left")

        # Detect float/int
        is_float = isinstance(var, tk.DoubleVar) or isinstance(min_val, float) or isinstance(max_val, float)

        # Create slider
        slider = ttk.Scale(frame, from_=min_val, to=max_val, orient="horizontal",
                        variable=var, length=160)
        slider.pack(side="left", padx=(6, 0), fill="x", expand=True)

        val_label = tk.Label(frame, text=f"{var.get():.2f}" if is_float else f"{int(var.get())}")
        val_label.pack(side="left", padx=(6, 0))

        def _update_val(*args):
            try:
                val = var.get()
                val_label.config(text=f"{val:.2f}" if is_float else f"{int(val)}")
                if callback:
                    callback()
            except Exception as e:
                print(f"[WARN] make_param_slider update failed: {e}")

        var.trace_add("write", _update_val)
        return slider

    def _mv_convert_detected(self, attr_name: str):
        """
        Convert detected shapes into bounding boxes and push them into Extras.
        Adds clipping to image bounds and avoids redundant re-conversion.
        """
        import numpy as np

        # print(f"[DEBUG] >>> ENTERED _mv_convert_detected with attr_name={attr_name}")
        # print(f"[DEBUG] mv_detections keys currently: {list(self.mv_detections.keys())}")

        if getattr(self, "_mv_in_conversion", False):
            # print("[DEBUG] Conversion already running; skipping.")
            return
        self._mv_in_conversion = True

        shape_key = attr_name.replace("detected_", "")
        allowed = {"circles", "circularity", "rects", "triangles", "polygons"}
        if shape_key not in allowed:
            print(f"[INFO] Conversion not allowed for {attr_name}")
            self._mv_in_conversion = False
            return

        if not hasattr(self, "mv_detections") or not isinstance(self.mv_detections, dict):
            print("[ERROR] self.mv_detections missing")
            self._mv_in_conversion = False
            return

        # Skip if already converted
        if self.mv_detections.get(f"{shape_key}_bboxes"):
            # print(f"[DEBUG] Skipping redundant conversion for {shape_key}")
            self._mv_in_conversion = False
            return

        items = self.mv_detections.get(shape_key, [])
        # print(f"[DEBUG] Retrieved {len(items)} items from mv_detections[{shape_key}]")
        if len(items):
            # print(f"[DEBUG] First few items: {items[:3]}")
            pass

        # Get image bounds for clipping
        if getattr(self, "current_image", None):
            W, H = self.current_image.size
        else:
            W = H = None

        # (Re)start local list
        self.mv_bboxes = []
        added = 0

        def _clip(v, lo, hi):
            return max(lo, min(hi, v))

        if shape_key == "circles":
            for entry in items:
                try:
                    if isinstance(entry, np.ndarray):
                        entry = entry.flatten().tolist()
                    if not (isinstance(entry, (list, tuple)) and len(entry) >= 3):
                        print(f"[WARN] Unexpected circle entry format: {entry}")
                        continue

                    x, y, r = map(int, entry[:3])
                    x1, y1, x2, y2 = x - r, y - r, x + r, y + r

                    # Clip to image bounds if available
                    if W is not None and H is not None:
                        x1 = _clip(x1, 0, W - 1)
                        y1 = _clip(y1, 0, H - 1)
                        x2 = _clip(x2, 0, W - 1)
                        y2 = _clip(y2, 0, H - 1)

                    if x2 <= x1 or y2 <= y1:
                        continue  # fully out of bounds after clipping

                    self.add_bbox_extra(x1, y1, x2, y2, label="circle")
                    self.mv_bboxes.append((x1, y1, x2, y2, "circle"))
                    added += 1
                    # print(f"[DEBUG] Added bbox for circle ({x},{y},{r}) ? ({x1},{y1},{x2},{y2})")

                except Exception as e:
                    print(f"[ERROR] Could not convert circle entry {entry}: {e}")
        elif shape_key == "circularity":
            for entry in items:
                try:
                    if isinstance(entry, (list, tuple)) and len(entry) >= 3:
                        x, y, r = map(int, entry[:3])
                        x1, y1, x2, y2 = x - r, y - r, x + r, y + r
                        self.add_bbox_extra(x1, y1, x2, y2, label="circularity")
                        self.mv_bboxes.append((x1, y1, x2, y2, "circularity"))
                        # print(f"[DEBUG] Added bbox for circularity ({x}, {y}, {r}) ? ({x1}, {y1}, {x2}, {y2})")
                        added += 1
                except Exception as e:
                    print(f"[ERROR] Could not convert circularity entry {entry}: {e}")

        elif shape_key == "rects":
            for entry in items:
                try:
                    if isinstance(entry, np.ndarray):
                        entry = entry.flatten().tolist()
                    if len(entry) >= 4:
                        x, y, w, h = map(int, entry[:4])
                        x1, y1, x2, y2 = x, y, x + w, y + h
                        if W is not None and H is not None:
                            x1 = _clip(x1, 0, W - 1); y1 = _clip(y1, 0, H - 1)
                            x2 = _clip(x2, 0, W - 1); y2 = _clip(y2, 0, H - 1)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        self.add_bbox_extra(x1, y1, x2, y2, label="rect")
                        self.mv_bboxes.append((x1, y1, x2, y2, "rect"))
                        added += 1
                except Exception as e:
                    print(f"[ERROR] Could not convert rect entry {entry}: {e}")

        elif shape_key in ("triangles", "polygons"):
            for pts in items:
                try:
                    if isinstance(pts, np.ndarray):
                        pts = pts.reshape(-1, 2).tolist()
                    if not pts:
                        continue
                    xs, ys = zip(*pts)
                    x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
                    if W is not None and H is not None:
                        x1 = _clip(int(x1), 0, W - 1); y1 = _clip(int(y1), 0, H - 1)
                        x2 = _clip(int(x2), 0, W - 1); y2 = _clip(int(y2), 0, H - 1)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    self.add_bbox_extra(x1, y1, x2, y2, label=shape_key)
                    self.mv_bboxes.append((x1, y1, x2, y2, shape_key))
                    added += 1
                except Exception as e:
                    print(f"[ERROR] Could not convert polygon entry {pts}: {e}")

        if added > 0:
            self.mv_detections[f"{shape_key}_bboxes"] = list(self.mv_bboxes)
            # print(f"[DEBUG] Stored {added} bounding boxes in mv_detections['{shape_key}_bboxes']")
        else:
            print(f"[WARN] No bounding boxes found for {shape_key}")

        print(f"[INFO] Extracted {added} bounding boxes from {shape_key}")

        # Make sure Extras are visible
        try:
            if hasattr(self, "show_extra"):
                self.show_extra.set(True)
        except Exception:
            pass

        self._mv_in_conversion = False
        self._render_image_on_canvas()

    def _mv_run_and_maybe_add(self, shape_type):
        """
        Run a shape detection preview and automatically convert detections to bounding boxes.
        Prevents redundant conversion loops.
        """
        shape_type = (shape_type or "").strip().lower()
        shapes = list(self.shape_dispatch.keys())
        # Normalize names
        if shape_type in ["hough circles", "circle", "circles"]:
            shape_type = "circles"
        elif shape_type in ["rect", "rects", "square", "squares"]:
            shape_type = "rects"
        elif shape_type in ["triangle", "triangles"]:
            shape_type = "triangles"
        elif shape_type in ["polygon", "polygons"]:
            shape_type = "polygons"
        elif shape_type in ["contour", "contours"]:
            shape_type = "contours"
        elif shape_type in ["circularity", "circularity detection", "roundness"]:
            shape_type = "circularity"


        # print(f"[DEBUG] >>> ENTERED _mv_run_and_maybe_add() for shape_type={shape_type}")

        # Run detection first
        self._mv_run_preview(shape_type)

        # --- Handle circles ---
        if shape_type == "circles":
            # print("[DEBUG] Handling circles conversion logic...")
            # Only convert once per fresh detection
            if "circles_bboxes" not in self.mv_detections:
                # print("[DEBUG] Calling _mv_convert_detected() for circles (first time)")
                self._mv_convert_detected("detected_circles")
            else:
                # print("[DEBUG] Skipping conversion - circles_bboxes already exist.")
                pass
            return


        # --- Handle other shapes (rects, triangles, etc.) ---
        if shape_type in ("rects", "triangles", "polygons", "contours"):
            if getattr(self, "mv_auto_add_shapes", None) and self.mv_auto_add_shapes.get():
                # print(f"[DEBUG] Auto-convert enabled for {shape_type}")
                self._mv_convert_detected(f"detected_{shape_type}")

        # print(f"[DEBUG] After convert: total extras = {len(getattr(self, 'mv_bboxes', []))}")

    def parse_selected_class(self):
        """
        Robustly parse the class index selected in the Combobox.
        Supports values like "3", "3: name", or "3: (New class)".
        Returns an int (>=0). For "New class" entry returns the new index (== total_classes).
        """
        sel = None
        try:
            sel = self.class_selector.get()
        except Exception:
            sel = getattr(self, "cls_var", "0")
        # If it's already an int var get()
        try:
            if isinstance(sel, int):
                return int(sel)
        except Exception:
            pass

        if not sel:
            return 0
        # if format "i: name" -> take before ':'
        if ":" in sel:
            left = sel.split(":", 1)[0].strip()
            try:
                return int(left)
            except Exception:
                pass
        # if plain numeric
        try:
            return int(sel)
        except Exception:
            # fallback: try to find trailing number
            import re
            m = re.search(r"\d+", sel)
            if m:
                return int(m.group(0))
        return 0

    def parse_selected_seg_class(self):
        """Parse the class index from the Segmentation class selector robustly.
        Accepts values like "3", or "3: name". Returns an int >= 0.
        """
        sel = None
        try:
            sel = self.seg_class_selector.get()
        except Exception:
            try:
                sel = self.seg_cls_var.get()
            except Exception:
                sel = "0"
        if not sel:
            return 0
        try:
            if isinstance(sel, int):
                return int(sel)
        except Exception:
            pass
        if ":" in str(sel):
            left = str(sel).split(":", 1)[0].strip()
            try:
                return int(left)
            except Exception:
                pass
        try:
            return int(str(sel).strip())
        except Exception:
            import re
            m = re.search(r"\d+", str(sel))
            if m:
                return int(m.group(0))
        # Fallback to bbox selector parsing
        try:
            return int(self.parse_selected_class())
        except Exception:
            return 0
    
    def clear_mv_preview(self):
        """Remove all Machine Vision overlays (shapes and previews) from canvas."""
        # Clear detected shapes
        for attr in ("detected_circles", "detected_rects", "detected_triangles", "detected_polygons"):
            if hasattr(self, attr):
                setattr(self, attr, [])
        # Clear the MV preview layer (but keep mask and index intact)
        self.mv_preview_image = None
        if hasattr(self, "show_mv_preview"):
            self.show_mv_preview.set(False)
        if hasattr(self, "_render_image_on_canvas"):
            self._render_image_on_canvas()
    
    def update_slider_range(self):
        n = max(1, len(self.image_files))
        self.slider.config(to=max(0, n - 1))
        if self.image_files:
            pos = int(self.current_index)
            self.slider_var.set(pos)
        if not self.image_files or self.current_index < 0:
            self.current_image = None
            self.current_image_path = None  # also reset it
            return

        img_path = self.image_files[self.current_index]
        try:
            img = Image.open(img_path).convert("RGB")
            self.current_image = img
            self.current_image_path = img_path  # ? store it here
            self.img_w, self.img_h = img.size
            # print(f"[DEBUG] Loaded image: {img_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open image:\n{img_path}\n{e}")
            self.current_image = None
            self.current_image_path = None

    def show_image_at_index(self):
        """Show image at self.current_index and update UI elements (navigation + dataset info)."""
        import os

        # print(f"[DEBUG] show_image_at_index() | index={getattr(self, 'current_index', None)} | total={len(getattr(self, 'image_files', []))}")

        # --- If no images are loaded ---
        if not getattr(self, "image_files", None) or self.current_index < 0:
            if hasattr(self, "lbl_nav_image_name"):
                self.lbl_nav_image_name.config(text="No image")
            if hasattr(self, "lbl_image_name"):
                self.lbl_image_name.config(text="No image loaded")

            self.canvas.delete("all")
            self.slider.config(to=0)

            # Reset info labels (optional)
            if hasattr(self, "lbl_shape_info"):
                self.lbl_shape_info.config(text="Shape: -")
            if hasattr(self, "lbl_counts_info"):
                self.lbl_counts_info.config(text="GT: 0   Pred: 0   Extra: 0")
            if hasattr(self, "lbl_class_count"):
                self.lbl_class_count.config(text="Classes: -")
            return

        # --- Clamp index ---
        self.current_index = max(0, min(self.current_index, len(self.image_files) - 1))
        self.load_current_image()
        current_img_path = self.image_files[self.current_index]
        if getattr(self, "selected_seg_mask_image", None) != current_img_path:
            self.selected_seg_mask = None
            self.selected_seg_mask_image = None

        # --- Reset old MV results ---
        self.mv_preview_image = None
        self.detected_circles = []
        self.detected_rects = []
        self.detected_triangles = []
        self.detected_polygons = []
        self.mv_tk_image = None

        # --- Update image name labels ---
        img_name = os.path.basename(self.image_files[self.current_index])
        total_imgs = len(self.image_files)

        # ? Navigation bar label (above slider)
        if hasattr(self, "lbl_nav_image_name"):
            self.lbl_nav_image_name.config(
                text=f"{self.current_index + 1}/{total_imgs}  –  {img_name}"
            )

        # ? Dataset info panel label (“Current Image” section)
        if hasattr(self, "lbl_image_name"):
            self.lbl_image_name.config(text=img_name)

        # --- Update dataset/image info stats ---
        try:
            img_w, img_h = self.img_w, self.img_h
            gt_count = len(self.bboxes_gt.get(self.image_files[self.current_index], []))
            pred_count = len(self.bboxes_pred.get(self.image_files[self.current_index], []))
            extra_count = len(self.bboxes_extra.get(self.image_files[self.current_index], []))

            # Shape info
            if hasattr(self, "lbl_shape_info"):
                self.lbl_shape_info.config(text=f"Shape: {img_w} x {img_h}")

            # Object counts
            if hasattr(self, "lbl_counts_info"):
                self.lbl_counts_info.config(
                    text=f"GT: {gt_count}   Pred: {pred_count}   Extra: {extra_count}"
                )

            # Unique class count
            all_boxes = (
                self.bboxes_gt.get(self.image_files[self.current_index], []) +
                self.bboxes_pred.get(self.image_files[self.current_index], []) +
                self.bboxes_extra.get(self.image_files[self.current_index], [])
            )
            unique_classes = {getattr(bb, "cls", None) for bb in all_boxes if getattr(bb, "cls", None) is not None}

            if hasattr(self, "lbl_class_count"):
                self.lbl_class_count.config(text=f"Classes: {len(unique_classes)}")

            # Optional debug detected shapes
            dc = getattr(self, "detected_circles", []) or []
            dr = getattr(self, "detected_rects", []) or []
            dt = getattr(self, "detected_triangles", []) or []
            dp = getattr(self, "detected_polygons", []) or []
            if hasattr(self, "lbl_detected"):
                self.lbl_detected.config(
                    text=f"Detected – Circles: {len(dc)}  Rects: {len(dr)}  Tris: {len(dt)}  Polys: {len(dp)}"
                )

        except Exception as e:
            print(f"[WARN] show_image_at_index() data update failed: {e}")

        # --- Reset offset & render image ---
        self._preserve_offset = False
        self.update_slider_range()
        self._render_image_on_canvas()

    def refresh_dataset_info(self):
        """Compute dataset-level stats from YOLO labels and update labels."""
        import os
        from collections import Counter

        label_folder = getattr(self, "label_folder", None)
        image_files = getattr(self, "image_files", [])
        if not label_folder or not os.path.isdir(label_folder):
            self.dataset_info = {"total_images": len(image_files)}
            if hasattr(self, "update_dataset_stats"):
                self.update_dataset_stats()
            return

        total_annotations = 0
        class_counter = Counter()

        for fname in os.listdir(label_folder):
            if not fname.endswith(".txt"):
                continue
            try:
                with open(os.path.join(label_folder, fname), "r", encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        try:
                            cls = int(parts[0])
                            class_counter[cls] += 1
                            total_annotations += 1
                        except Exception:
                            continue
            except Exception:
                continue

        total_images = len(image_files)
        avg_boxes = total_annotations / total_images if total_images else 0
        balance = ", ".join([f"{cid} ({n})" for cid, n in class_counter.most_common(3)]) if class_counter else "-"

        self.dataset_info = {
            "total_images": total_images,
            "total_annotations": total_annotations,
            "avg_boxes": f"{avg_boxes:.2f}",
            "class_balance": balance,
        }

        if hasattr(self, "update_dataset_stats"):
            self.update_dataset_stats()

    # ---------- MV indices/threshold ----------
    def excessive_index_apply(self, index_name="ExG", preview_only=False, class_id=0):
        """Compute mask and optionally add bounding boxes to extras."""
        if not hasattr(self, "current_image") or not self.current_image:
            messagebox.showerror("Error", "No image loaded.")
            return

        mask = getattr(self, "mv_mask_image", None)
        if mask is None:
            messagebox.showerror("Error", f"No {index_name} mask available.")
            return

        self._render_image_on_canvas()

        if not preview_only:
            bbox = mask.getbbox()
            if not bbox:
                messagebox.showinfo("No mask", f"{index_name} mask produced no region.")
                return

            x1, y1, x2, y2 = bbox
            new_bb = BBox(class_id, x1, y1, x2, y2)
            img_path = self.image_files[self.current_index]
            self.bboxes_extra.setdefault(img_path, []).append(new_bb)
            messagebox.showinfo("Extra Added", f"Added a bbox from {index_name} mask to Extras.")
            self._render_image_on_canvas()

    def apply_threshold_to_index(self, index_name="ExG"):
        """Apply threshold to generated index image and update mask preview."""
        if not hasattr(self, "ex_index_image") or self.ex_index_image is None:
            self.generate_index_image(index_name)
            if not hasattr(self, "ex_index_image") or self.ex_index_image is None:
                messagebox.showerror("Error", f"No {index_name} image generated.")
                return

        thresh = self.mv_idx_thresh.get()
        try:
            mask = Image.new("L", self.ex_index_image.size)
            mask_px = mask.load()
            idx_px = self.ex_index_image.load()
            for y in range(self.ex_index_image.height):
                for x in range(self.ex_index_image.width):
                    mask_px[x, y] = 255 if idx_px[x, y] > thresh else 0
            mask = mask.filter(ImageFilter.MaxFilter(3))

            # ? store in mv_mask_image (not mv_preview_image)
            self.mv_mask_image = mask
            self.show_ex_mask.set(True)
            self._render_image_on_canvas()
        except Exception as e:
            print(f"[Error] Threshold failed: {e}")

    def _apply_threshold_to_index_fast(self, index_name="ExG"):
        """Vectorized thresholding for the current index image with preview update."""
        if not hasattr(self, "ex_index_image") or self.ex_index_image is None:
            self.generate_index_image(index_name)
            if not hasattr(self, "ex_index_image") or self.ex_index_image is None:
                messagebox.showerror("Error", f"No {index_name} image generated.")
                return

        try:
            thresh = int(self.mv_idx_thresh.get()) if hasattr(self, "mv_idx_thresh") else 50
            arr = np.array(self.ex_index_image, dtype=np.uint8)
            mask_arr = (arr > thresh).astype(np.uint8) * 255
            mask = Image.fromarray(mask_arr, mode="L").filter(ImageFilter.MaxFilter(3))

            self.mv_mask_image = mask
            if hasattr(self, "show_ex_mask"):
                self.show_ex_mask.set(True)
            # Build RGB*Mask preview for readability layer
            try:
                rgb = np.array(self.current_image.convert("RGB"), dtype=np.uint8)
                m3 = np.stack([mask_arr] * 3, axis=-1)
                rgb_masked = ((rgb.astype(np.uint16) * m3.astype(np.uint16)) // 255).astype(np.uint8)
                self.ex_rgb_mask_image = Image.fromarray(rgb_masked)
            except Exception:
                self.ex_rgb_mask_image = None
            self._render_image_on_canvas()
        except Exception as e:
            print(f"[Error] Fast threshold failed: {e}")

    def save_current_mask(self):
        """Save the current binary mask (mv_mask_image) to a chosen folder as PNG.
        Uses the active image filename with suffix _mask.png.
        """
        from tkinter import messagebox, filedialog
        import os
        try:
            # Validate mask and image context
            if not hasattr(self, "mv_mask_image") or self.mv_mask_image is None:
                messagebox.showerror("No Mask", "No binary mask to save. Threshold an index first.")
                return
            if not getattr(self, "image_files", None) or self.current_index < 0:
                messagebox.showerror("No Image", "No active image to name the mask file.")
                return

            # Ensure output folder
            out_dir = getattr(self, "mask_save_folder", None)
            if not out_dir:
                out_dir = filedialog.askdirectory(title="Select folder to save masks")
                if not out_dir:
                    return
                self.mask_save_folder = out_dir
                if hasattr(self, "mask_save_folder_var"):
                    try:
                        self.mask_save_folder_var.set(out_dir)
                    except Exception:
                        pass

            base = os.path.splitext(os.path.basename(self.image_files[self.current_index]))[0]
            out_path = os.path.join(out_dir, f"{base}_mask.png")

            # Save as 8-bit grayscale
            mask_img = self.mv_mask_image
            try:
                if getattr(mask_img, 'mode', None) != 'L':
                    mask_img = mask_img.convert('L')
                mask_img.save(out_path)
                messagebox.showinfo("Saved", f"Saved mask to:\n{out_path}")
            except Exception as e:
                messagebox.showerror("Save Failed", f"Could not save mask to:\n{out_path}\n{e}")
        except Exception as e:
            print(f"[ERROR] save_current_mask: {e}")

    def generate_index_image(self, index_name="ExG"):
        """
        Generate a general index image from R,G,B channels.
        Also clears any Machine Vision shape preview to restore Step 2 view.
        """
        if not hasattr(self, "current_image") or not self.current_image:
            messagebox.showerror("Error", "No image loaded.")
            return

        try:
            im = self.current_image.convert("RGB")
            r, g, b = [np.asarray(ch, dtype=np.float32) for ch in im.split()]

            name_in = (index_name or "").strip()
            key = name_in.lower().replace(" ", "")
            if key == "customrgb":
                ar = float(self.custom_r_coef.get()) if hasattr(self, "custom_r_coef") else 1.0
                ag = float(self.custom_g_coef.get()) if hasattr(self, "custom_g_coef") else 0.0
                ab = float(self.custom_b_coef.get()) if hasattr(self, "custom_b_coef") else 0.0
                index = ar * r + ag * g + ab * b
            else:
                if name_in not in self.INDEX_FORMULAS:
                    messagebox.showerror("Error", f"Index {name_in} not supported.")
                    return
                formula = self.INDEX_FORMULAS[name_in]
                index = formula(r, g, b)

            # ?? Normalize 0–255 for display
            index_norm = cv2.normalize(index, None, 0, 255, cv2.NORM_MINMAX)
            index_im = Image.fromarray(index_norm.astype(np.uint8), mode="L")

            # ?? Store the grayscale index image
            self.ex_index_image = index_im

            # ? Build RGB × Index composite for visualization
            im_np = np.asarray(im, dtype=np.uint8)
            index_3ch = cv2.merge([index_norm] * 3)
            mult = np.clip(im_np.astype(np.float32) + index_3ch * 0.5, 0, 255).astype(np.uint8)
            self.ex_index_mult_image = Image.fromarray(mult)

            # ?? Clear any existing MV shape preview (from Step 3)
            if hasattr(self, "clear_mv_preview"):
                self.clear_mv_preview()
            else:
                # fallback if helper missing
                self.mv_preview_image = None
                if hasattr(self, "show_mv_preview"):
                    self.show_mv_preview.set(False)
                for attr in ("detected_circles", "detected_rects", "detected_triangles", "detected_polygons"):
                    if hasattr(self, attr):
                        setattr(self, attr, [])

            # ?? Re-render updated index & mask layers
            self._render_image_on_canvas()

        except Exception as e:
            print(f"Error generating {index_name}:", e)
            messagebox.showerror("Error", f"Failed to generate {index_name}.")

    def _mv_run_preview(self, shape_type):
        """Run a preview of the selected shape detection (circles, rects, etc.)."""
        # Normalize
        shape_type = (shape_type or "").strip().lower()

        if shape_type in ["circles"]:
            shape_type = "circles"

        elif shape_type in ["rect", "rects", "square", "squares"]:
            shape_type = "rects"
        elif shape_type in ["triangle", "triangles"]:
            shape_type = "triangles"
        elif shape_type in ["polygon", "polygons"]:
            shape_type = "polygons"
        elif shape_type in ["contour", "contours"]:
            shape_type = "contour (generic)"
        elif shape_type in ["circularity", "roundness", "blob"]:
            shape_type = "circularity"
        # Ensure dispatch exists
        if not hasattr(self, "shape_dispatch") or not isinstance(self.shape_dispatch, dict):
            self.shape_dispatch = {}

        # ?? Ensure mv_detections exists before any detection
        if not hasattr(self, "mv_detections") or not isinstance(self.mv_detections, dict):
            self.mv_detections = {}
            # print("[DEBUG] Created self.mv_detections dictionary.")

        # Map shape detection functions
        from MV.Shapes import circle_detect, rectangle_detect, triangle_detect, polygon_detect, contour_detect_generic, contour_detect_mask

        self.shape_dispatch.update({
            "circles": circle_detect,                  # Hough-based circle detection
            "circularity": circle_detect_circularity,  # NEW: contour-based circularity
            "square detection": rectangle_detect,
            "triangle detection": triangle_detect,
            "polygon detection": polygon_detect,
            "contour (mask)": contour_detect_mask,
            "contour (generic)": contour_detect_generic,
        })


        img = self._get_mv_input_image()
        if img is None:
            print("[ERROR] No valid input image for detection.")
            return

        # print(f"[DEBUG] Input type to detection: {type(img)}")
        # print(f"[DEBUG] Image mode: {getattr(img, 'mode', None)}, size: {getattr(img, 'size', None)}")
        # print(f"[DEBUG] Running preview for shape type: {shape_type.upper()}")

        func = self.shape_dispatch.get(shape_type)
        if func:
            func(self, img)
        else:
            print(f"[WARN] Unknown shape type: {shape_type}")

        # ?? Confirm after detection
        # print(f"[DEBUG] After detection, mv_detections keys: {list(self.mv_detections.keys())}")

        self._render_image_on_canvas()

    def _mv_preview_bboxes(self, shape_name):
        """
        Compute bounding boxes for the detected shapes and draw them on top of the current preview.
        """
        import cv2
        import numpy as np
        from PIL import Image

        bboxes = self._mv_get_bboxes_from_shape(shape_name)
        if not bboxes:
            print(f"[WARN] No bounding boxes found for {shape_name}")
            return

        # print(f"[DEBUG] Previewing {len(bboxes)} bounding boxes from {shape_name}")

        # --- Draw the boxes over the current preview image ---
        if not hasattr(self, "mv_preview_image") or self.mv_preview_image is None:
            print("[WARN] No preview image available; cannot draw bboxes.")
            return

        img = np.array(self.mv_preview_image.convert("RGB"))
        for (x1, y1, x2, y2) in bboxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # green box

        self.mv_preview_image = Image.fromarray(img)
        self._render_image_on_canvas()

    def _get_mv_input_image(self):
        src = getattr(self, "mv_input_source", None)
        sel = (src.get() if src else "RGB").lower()

        if sel == "index" and hasattr(self, "ex_index_image") and self.ex_index_image is not None:
            return self.ex_index_image
        elif sel == "mask" and hasattr(self, "mv_mask_image") and self.mv_mask_image is not None:
            return self.mv_mask_image
        elif hasattr(self, "current_image") and self.current_image is not None:
            return self.current_image
        else:
            print(f"[WARN] No valid {sel} image found.")
            return None
    
    

    # ---------- Rendering ----------
    def _render_image_on_canvas(self, *args):
        """Render base image, overlays, and machine vision detected shapes."""
        self.canvas.delete("all")
        self.handle_id_to_info.clear()
        self.canvas.delete("highlight")

        if not self.current_image:
            return

        # Canvas size
        c_w, c_h = self.canvas.winfo_width() or 800, self.canvas.winfo_height() or 600
        if c_w < 10 or c_h < 10:
            return

        # compute display scale
        base_scale = self.compute_base_scale(c_w, c_h)
        self.display_scale = base_scale * self.zoom_factor

        disp_w, disp_h = max(1, int(self.img_w * self.display_scale)), max(1, int(self.img_h * self.display_scale))

        # center image unless preserving offset
        if not getattr(self, '_preserve_offset', False):
            self.offset_x = (c_w - disp_w) / 2.0
            self.offset_y = (c_h - disp_h) / 2.0

        # Resize base image
        try:
            img_resized = self.current_image.resize((disp_w, disp_h), Image.LANCZOS)
        except Exception:
            img_resized = self.current_image.copy().resize((disp_w, disp_h), Image.LANCZOS)
        base_rgba = img_resized.convert("RGBA")

        # Create overlay for PIL boxes
        overlay = Image.new("RGBA", (disp_w, disp_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay, "RGBA")

        # Collect boxes
        img_path = self.image_files[self.current_index] if self.image_files else None
        gt_boxes = self.bboxes_gt.get(img_path, []) if img_path else []
        pred_boxes = self.bboxes_pred.get(img_path, []) if img_path else []
        extra_boxes = []
        for p in pred_boxes:
            matched = any(
                calculate_iou(p.as_tuple(), g.as_tuple()) > 0.5
                for g in gt_boxes if hasattr(p, "as_tuple") and hasattr(g, "as_tuple")
            )
            if not matched:
                extra_boxes.append(p)
        extra_boxes.extend(self.bboxes_extra.get(img_path, []) if img_path else [])

        # Colors fallback
        if not hasattr(self, "class_colors") or not isinstance(self.class_colors, dict):
            self.class_colors = {0: (0, 200, 0), 1: (255, 80, 80), 2: (0, 140, 255),
                                3: (255, 200, 0), 4: (200, 0, 200)}

        def _get_class_and_coords(bb):
            try:
                cls = int(getattr(bb, "cls", 0))
                x1, y1, x2, y2 = float(bb.x1), float(bb.y1), float(bb.x2), float(bb.y2)
                return cls, x1, y1, x2, y2
            except Exception:
                if isinstance(bb, (list, tuple)) and len(bb) >= 4:
                    cls = int(bb[0]) if len(bb) >= 5 else 0
                    x1, y1, x2, y2 = map(float, bb[-4:])
                    return cls, x1, y1, x2, y2
            return 0, 0, 0, 0, 0

        # Draw PIL boxes for GT / Pred / Extra
        def draw_pil_boxes(box_list, trans_var):
            alpha = max(0.0, min(1.0, trans_var.get()))
            if alpha <= 0.0:
                return
            FILL_MAX, OUTLINE_MAX = 150, 220
            fill_a = int(round(alpha * FILL_MAX))
            outline_a = int(round(alpha * OUTLINE_MAX))
            for bb in box_list:
                cls, x1_i, y1_i, x2_i, y2_i = _get_class_and_coords(bb)
                x1 = max(0, int(x1_i * self.display_scale))
                y1 = max(0, int(y1_i * self.display_scale))
                x2 = min(disp_w, int(x2_i * self.display_scale))
                y2 = min(disp_h, int(y2_i * self.display_scale))
                color = self.class_colors.get(int(cls), (180, 180, 180))
                draw.rectangle([x1, y1, x2, y2],
                            fill=(color[0], color[1], color[2], fill_a),
                            outline=(color[0], color[1], color[2], outline_a), width=2)

        if self.show_gt.get(): draw_pil_boxes(gt_boxes, self.transparency_gt)
        if self.show_pred.get(): draw_pil_boxes(pred_boxes, self.transparency_pred)
        if self.show_extra.get(): draw_pil_boxes(extra_boxes, self.transparency_extra)

        # Draw segmentation masks as filled polygons (per-layer with alpha)
        def draw_seg_masks(mask_list, trans_var):
            alpha = float(getattr(trans_var, 'get', lambda: 0.0)())
            if alpha <= 0.0:
                return
            FILL_MAX, OUTLINE_MAX = 160, 230
            fill_a = int(round(alpha * FILL_MAX))
            outline_a = int(round(alpha * OUTLINE_MAX))
            for mk in mask_list or []:
                try:
                    cls = int(mk.get("cls", 0))
                    pts_img = mk.get("points", [])
                    if len(pts_img) < 3:
                        continue
                    color = self.class_colors.get(int(cls), (180, 180, 180))
                    pts_disp = []
                    for (px, py) in pts_img:
                        dx = max(0, min(disp_w, int(px * self.display_scale)))
                        dy = max(0, min(disp_h, int(py * self.display_scale)))
                        pts_disp.append((dx, dy))
                    draw.polygon(pts_disp, fill=(color[0], color[1], color[2], fill_a),
                                 outline=(color[0], color[1], color[2], outline_a))
                except Exception:
                    pass

        try:
            if getattr(self, 'seg_show_gt', None) and self.seg_show_gt.get():
                draw_seg_masks(self.seg_masks_gt.get(img_path, []), self.seg_transparency_gt)
            if getattr(self, 'seg_show_pred', None) and self.seg_show_pred.get():
                draw_seg_masks(self.seg_masks_pred.get(img_path, []), self.seg_transparency_pred)
            if getattr(self, 'seg_show_extra', None) and self.seg_show_extra.get():
                draw_seg_masks(self.seg_masks_extra.get(img_path, []), self.seg_transparency_extra)
        except Exception:
            pass

        combined = Image.alpha_composite(base_rgba, overlay)

        # Ensure MV/index layers are PIL Images (not numpy arrays) before resizing
        try:
            import numpy as _np
            from PIL import Image as _PILImage
            for _attr in ("ex_index_image", "mv_preview_image", "ex_rgb_mask_image"):
                if hasattr(self, _attr):
                    _src = getattr(self, _attr)
                    if isinstance(_src, _np.ndarray):
                        try:
                            setattr(self, _attr, _PILImage.fromarray(_src.copy()))
                        except Exception:
                            pass
        except Exception:
            pass

        # --- Machine Vision layers ---
        try:
            disp_size = (disp_w, disp_h)
            for name, attr, show_var, trans_var in [
                ("Index", "ex_index_image", self.show_ex_index, self.transparency_ex_index),
                ("Mask", "mv_mask_image", self.show_ex_mask, self.transparency_ex_mask),
                ("RGBMask", "ex_rgb_mask_image", self.show_ex_index_mult, self.transparency_ex_index_mult),
                # Only blend MV preview if no shapes are drawn
                ("MVPreview", "mv_preview_image", self.show_mv_preview, self.transparency_mv_preview)
            ]:

                if hasattr(self, attr) and show_var.get():
                    src = getattr(self, attr, None)
                    if src is None:
                        continue
                    try:
                        import numpy as _np
                        from PIL import Image as _PILImage
                        if isinstance(src, _np.ndarray):
                            src = _PILImage.fromarray(src.copy())
                    except Exception:
                        pass
                    try:
                        layer = src.resize(disp_size).convert("RGBA")
                        alpha = int(255 * trans_var.get())
                        layer.putalpha(alpha)
                        combined = Image.alpha_composite(combined, layer)
                    except Exception as _e:
                        print("Error rendering index layer:", name, _e)

        except Exception as e:
            print("[ERROR] Error rendering index layers:", e)

        # Show base + overlays
        self.tk_image = ImageTk.PhotoImage(combined)
        self.canvas.create_image(self.offset_x, self.offset_y, anchor=tk.NW,
                                image=self.tk_image, tags=("base_img",))

        # Highlight currently selected bbox (outline + handles)
        try:
            if getattr(self, 'selected_bbox', None):
                bb = self.selected_bbox
                x1 = int(bb.x1 * self.display_scale + self.offset_x)
                y1 = int(bb.y1 * self.display_scale + self.offset_y)
                x2 = int(bb.x2 * self.display_scale + self.offset_x)
                y2 = int(bb.y2 * self.display_scale + self.offset_y)
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="#FFD400", width=2, tags=("highlight",))
            self._draw_handles(bb)
        except Exception:
            pass

        # Highlight selected segmentation mask (canvas overlay)
        try:
            if getattr(self, 'selected_seg_mask', None):
                current_img_path = self.image_files[self.current_index] if self.image_files else None
                if getattr(self, 'selected_seg_mask_image', None) != current_img_path:
                    # Drop stale selection from another image
                    self.selected_seg_mask = None
                    self.selected_seg_mask_image = None
                else:
                    mk = self.selected_seg_mask
                    pts = []
                    for (ix, iy) in mk.get("points", []):
                        cx = int(ix * self.display_scale + self.offset_x)
                        cy = int(iy * self.display_scale + self.offset_y)
                        pts.extend([cx, cy])
                    if len(pts) >= 6:
                        self.canvas.create_polygon(*pts, outline="#FFD400", fill="", width=2, tags=("highlight",))
                        if self.current_mode.get() == "modify shape":
                            # Draw vertex handles only in modify mode
                            try:
                                sz = max(3, int(self.HANDLE_SIZE))
                            except Exception:
                                sz = 6
                            pcoords = list(zip(pts[0::2], pts[1::2]))
                            for (cx, cy) in pcoords:
                                self.canvas.create_rectangle(
                                    cx - sz, cy - sz, cx + sz, cy + sz,
                                    outline="#FFD400", fill="#FFD400", width=1, tags=("highlight",))
        except Exception:
            pass

        # --- Draw detected shapes directly on canvas ---
        # Render vector shape overlays when MV Preview is enabled, independent of preview image existence
        if self.show_mv_preview.get():
            # Circles
            if hasattr(self, "detected_circles"):
                for x, y, r in self.detected_circles:
                    x_disp = int(x * self.display_scale + self.offset_x)
                    y_disp = int(y * self.display_scale + self.offset_y)
                    r_disp = int(r * self.display_scale)
                    self.canvas.create_oval(
                        x_disp - r_disp, y_disp - r_disp,
                        x_disp + r_disp, y_disp + r_disp,
                        outline="red", width=2, tags="mv_overlay"
                    )

            # Rectangles
            if hasattr(self, "detected_rects"):
                for x1, y1, x2, y2 in self.detected_rects:
                    x1_disp = int(x1 * self.display_scale + self.offset_x)
                    y1_disp = int(y1 * self.display_scale + self.offset_y)
                    x2_disp = int(x2 * self.display_scale + self.offset_x)
                    y2_disp = int(y2 * self.display_scale + self.offset_y)
                    self.canvas.create_rectangle(
                        x1_disp, y1_disp, x2_disp, y2_disp,
                        outline="lime", width=2, tags="mv_overlay"
                    )

            # Draw triangles (actual contour)
            if hasattr(self, "detected_triangles") and self.show_mv_preview.get():
                for points in self.detected_triangles:
                    # --- Normalize points structure ---
                    clean_points = []
                    if isinstance(points, (list, tuple)):
                        for p in points:
                            if isinstance(p, (list, tuple)):
                                # Normal (x,y)
                                if len(p) == 2:
                                    clean_points.append(p)
                                # Circle-like or rect-like (x,y,r) / (x1,y1,x2,y2)
                                elif len(p) == 3:
                                    clean_points.append((p[0], p[1]))
                                elif len(p) == 4:
                                    clean_points.append((p[0], p[1]))
                                    clean_points.append((p[2], p[3]))
                    points = clean_points
                    if len(points) < 3:
                        continue

                    pts = [(int(x * self.display_scale + self.offset_x),
                            int(y * self.display_scale + self.offset_y)) for x, y in points]
                    self.canvas.create_polygon(pts, outline="cyan", fill="", width=2, tags="mv_overlay")

            # Draw polygons (actual contour)
            if hasattr(self, "detected_polygons") and self.show_mv_preview.get():
                for points in self.detected_polygons:
                    pts = [(int(x * self.display_scale + self.offset_x),
                            int(y * self.display_scale + self.offset_y)) for x, y in points]

                    self.canvas.create_polygon(pts, outline="orange", fill="", width=2, tags="mv_overlay")

            # --- Contour detection results ---
            if hasattr(self, "detected_contours") and self.detected_contours and self.show_mv_preview.get():
                draw_cv = ImageDraw.Draw(combined, "RGBA")
                for contour in self.detected_contours:
                    pts = [(int(x * self.display_scale), int(y * self.display_scale)) for (x, y) in contour]
                    if len(pts) > 1:
                        draw_cv.line(pts + [pts[0]], fill=(255, 255, 0, 180), width=2)

    # ---------- Mouse interactions ----------
    def canvas_to_image_coords(self, cx, cy):
        ix = (cx - self.offset_x) / max(1e-6, self.display_scale)
        iy = (cy - self.offset_y) / max(1e-6, self.display_scale)
        ix = max(0, min(self.img_w, ix))
        iy = max(0, min(self.img_h, iy))
        return ix, iy
    
    def _on_mousewheel_zoom(self, event):
        """Zoom in/out at the cursor position using the mouse wheel."""
        # Normalize scroll direction across OSs
        if event.num == 5 or event.delta < 0:
            factor = 0.7   # zoom out
        else:
            factor = 1.3   # zoom in

        # Get cursor coordinates on the canvas
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)

        # Call your existing zoom_at_cursor method
        self.zoom_at_cursor(cx, cy, factor)
    
    def _on_pan_start(self, event):
        """Record the start position for panning."""
        self._pan_start_x = event.x
        self._pan_start_y = event.y

    def _on_pan_move(self, event):
        """Move the image as the mouse drags."""
        dx = event.x - self._pan_start_x
        dy = event.y - self._pan_start_y
        self._pan_start_x = event.x
        self._pan_start_y = event.y

        # Pan by pixel deltas (canvas coords)
        self.pan_pixels(dx, dy)
    
    def on_mouse_down(self, event):
        x, y = event.x, event.y
        mode = self.current_mode.get()
        img_path = self.image_files[self.current_index] if self.image_files else None
        # Detect clicked bbox (respecting visibility)
        clicked_bbox, layer = self.find_bbox_at(x, y)
        # If segmentation tab is active, handle masks first
        if self._active_left_tab.get() == "seg":
            mk, mk_layer = self.find_seg_mask_at(x, y)
            if mode == "draw":
                self._seg_on_click_add_point(x, y)
                return
            if mode == "delete" and mk:
                try:
                    if messagebox.askyesno("Delete Mask", "Delete selected segmentation mask?"):
                        if mk_layer == "gt":
                            self.seg_masks_gt.get(img_path, []).remove(mk)
                        elif mk_layer == "pred":
                            self.seg_masks_pred.get(img_path, []).remove(mk)
                        else:
                            self.seg_masks_extra.get(img_path, []).remove(mk)
                        if self.selected_seg_mask is mk:
                            self.selected_seg_mask = None
                            self.selected_seg_mask_image = None
                        self._render_image_on_canvas()
                except Exception:
                    pass
                return
            if mode == "validate" and mk and mk_layer in ("pred", "extra"):
                try:
                    if mk_layer == "pred":
                        self.seg_masks_pred.get(img_path, []).remove(mk)
                    else:
                        self.seg_masks_extra.get(img_path, []).remove(mk)
                    self.seg_masks_gt.setdefault(img_path, []).append(mk)
                    self.selected_seg_mask = mk
                    self.selected_seg_mask_image = img_path
                    self._render_image_on_canvas()
                except Exception:
                    pass
                return
            if mode == "class change" and mk:
                self.selected_seg_mask = mk
                self.selected_seg_mask_image = img_path
                self.drag_data = {
                    "mode": "seg_class_change",
                    "mask": mk,
                    "layer": mk_layer,
                    "start": (x, y),
                }
                self._render_image_on_canvas()
                return
            # Move whole mask in Move/Resize mode
            if mk and mode in ("move", "resize"):
                self.selected_seg_mask = mk
                self.selected_seg_mask_image = img_path
                self.drag_data = {
                    "mode": "seg_move",
                    "mask": mk,
                    "start": (x, y),
                    "layer": mk_layer,
                }
                return
            # Allow dragging a single vertex only in Modify Shape mode
            if mode == "modify shape":
                vmk, vlayer, vidx = self.find_seg_vertex_at(x, y)
                if vmk is not None and vidx is not None:
                    self.selected_seg_mask = vmk
                    self.selected_seg_mask_image = img_path
                    self.drag_data = {
                        "mode": "seg_vertex",
                        "mask": vmk,
                        "vertex_index": int(vidx),
                        "layer": vlayer,
                    }
                    self._render_image_on_canvas()
                    return
            if mode in ("move", "resize"):
                self.selected_seg_mask = None
                self.selected_seg_mask_image = None
                self._render_image_on_canvas()
        # --- DRAW MODE ---
        if mode == "draw":
            self._drawing_mode = True
            self._new_rect_start = (x, y)
            if self._new_rect_id:
                try:
                    self.canvas.delete(self._new_rect_id)
                except Exception:
                    pass
            self._new_rect_id = self.canvas.create_rectangle(
                x, y, x, y, outline="#00CCCC", width=2, dash=(3, 3)
            )
            self.drag_data = {"mode": "draw", "start": (x, y)}
            return
        # --- CHANGE CLASS MODE ---
        if mode == "class change":
            if clicked_bbox:
                self.selected_bbox = clicked_bbox
                self.drag_data = {
                    "mode": "class change",
                    "start": (x, y),
                    "bbox": clicked_bbox,
                    "layer": layer,
                }
                # Highlight the selected bbox
                self._highlight_selected_bbox(clicked_bbox)
                print(f"[INFO] Selected bbox from layer '{layer}' for class change.")
            else:
                print("[INFO] No bbox selected for class change.")
            return
        # --- MOVE / DELETE / VALIDATE handled as before ---
        if mode == "delete" and clicked_bbox:
            if layer == "gt" and messagebox.askyesno("Delete", "Delete selected bbox?"):
                self.bboxes_gt[img_path].remove(clicked_bbox)
                self._render_image_on_canvas()
            return
        if mode == "validate" and clicked_bbox:
            if layer in ["pred", "extra"]:
                self.bboxes_gt.setdefault(img_path, []).append(clicked_bbox)
                if layer == "pred":
                    self.bboxes_pred[img_path].remove(clicked_bbox)
                else:
                    self.bboxes_extra[img_path].remove(clicked_bbox)
                self._render_image_on_canvas()
            return
        # --- SELECT / MOVE DEFAULT ---
        if clicked_bbox:
            self.selected_bbox = clicked_bbox
            self.drag_data = {
                "mode": "move",
                "start": (x, y),
                "bbox": clicked_bbox,
                "layer": layer,
            }
        else:
            self.selected_bbox = None
        self._render_image_on_canvas()
    
    def on_mouse_up(self, event):
        x, y = event.x, event.y
        mode = self.drag_data.get("mode")
        img_path = self.image_files[self.current_index] if self.image_files else None
        # Segmentation class change finalize
        if mode == "seg_class_change":
            mk = self.drag_data.get("mask")
            if not mk:
                self.drag_data = {"mode": None}
                return
            try:
                selected_class = int(self.parse_selected_seg_class())
            except Exception:
                selected_class = 0
            if not messagebox.askyesno("Change Class", f"Change mask class to {selected_class}?"):
                self.drag_data = {"mode": None}
                return
            mk["cls"] = selected_class
            self.update_dataset_info()
            self._render_image_on_canvas()
            self.drag_data = {"mode": None}
            return
        # --- CLASS CHANGE ---
        if mode == "class change":
            bbox = self.drag_data.get("bbox")
            layer = self.drag_data.get("layer")
            if bbox is None or layer is None:
                self.drag_data = {"mode": None}
                return
            # Verify layer still visible
            if (layer == "gt" and not self.show_gt.get()) or \
                (layer == "pred" and not self.show_pred.get()) or \
                (layer == "extra" and not self.show_extra.get()):
                print(f"[WARN] Layer '{layer}' hidden ? skipping class change.")
                self.drag_data = {"mode": None}
                return
            # Read selected class safely
            try:
                selected_class = self.parse_selected_class()
            except Exception:
                selected_class = 0
            # Confirmation dialog
            if not messagebox.askyesno("Change Class", f"Change class to {selected_class}?"):
                self.drag_data = {"mode": None}
                return
            # Update class
            if hasattr(bbox, "class_id"):
                bbox.class_id = selected_class
            elif hasattr(bbox, "cls"):
                bbox.cls = selected_class
            print(f"[INFO] Changed class to {selected_class} in layer '{layer}'")
            # Update dataset info (for new class addition)
            self.update_dataset_info()
            self._render_image_on_canvas()
            self.drag_data = {"mode": None}
            return
        # --- DRAW MODE ---
        if mode == "draw":
            x1, y1 = self._new_rect_start
            x2, y2 = event.x, event.y
            if self._new_rect_id:
                try:
                    self.canvas.delete(self._new_rect_id)
                except Exception:
                    pass
                self._new_rect_id = None
            ix1, iy1 = self.canvas_to_image_coords(x1, y1)
            ix2, iy2 = self.canvas_to_image_coords(x2, y2)
            a1, a2 = min(ix1, ix2), min(iy1, iy2)
            b1, b2 = max(ix1, ix2), max(iy1, iy2)
            if abs(b1 - a1) < 3 or abs(b2 - a2) < 3:
                self.drag_data = {"mode": None}
                self._drawing_mode = False
                return
            try:
                selected_class = self.parse_selected_class()
            except Exception:
                selected_class = 0
            new_bbox = BBox(selected_class, a1, a2, b1, b2)
            self.bboxes_gt.setdefault(img_path, []).append(new_bbox)
            self.selected_bbox = new_bbox
            self._drawing_mode = False
            self.drag_data = {"mode": None}
            self.update_dataset_info()
            self._render_image_on_canvas()
            return
        # --- MOVE MODE ---
        if mode == "move":
            bbox = self.drag_data.get("bbox")
            sx, sy = self.drag_data.get("start")
            if not bbox:
                return
            dx = (x - sx) / max(1e-9, self.display_scale)
            dy = (y - sy) / max(1e-9, self.display_scale)
            self._apply_delta_to_bbox(bbox, dx, dy)
            self._clamp_bbox_to_image(bbox, self.img_w, self.img_h)
            self.drag_data["start"] = (x, y)
            self._render_image_on_canvas()
        if mode == "seg_move":
            # finalize move on mouse up (already applied during move)
            self.drag_data = {"mode": None}
            return
        if mode == "seg_vertex":
            # finalize vertex edit
            self.drag_data = {"mode": None}
            return
        # --- reset drag state ---
        self.drag_data = {"mode": None}

    def on_mouse_move(self, event):
        x, y = event.x, event.y
        mode = self.drag_data.get("mode")
        # Seg live preview of edge
        if self._active_left_tab.get() == "seg" and self._seg_is_drawing:
            self._seg_update_preview_edge(x, y)
        # Seg moving a mask
        if mode == "seg_move":
            mk = self.drag_data.get("mask")
            sx, sy = self.drag_data.get("start")
            if not mk:
                return
            ix, iy = self.canvas_to_image_coords(x, y)
            isx, isy = self.canvas_to_image_coords(sx, sy)
            dx = ix - isx
            dy = iy - isy
            try:
                for i in range(len(mk["points"])):
                    px, py = mk["points"][i]
                    mk["points"][i] = (max(0, min(self.img_w, px + dx)),
                                        max(0, min(self.img_h, py + dy)))
                self.drag_data["start"] = (x, y)
                self._render_image_on_canvas()
            except Exception:
                pass
            return
        if mode == "seg_vertex":
            mk = self.drag_data.get("mask")
            vidx = self.drag_data.get("vertex_index")
            if mk is None or vidx is None:
                return
            # Set the vertex to the cursor position in image coords
            ix, iy = self.canvas_to_image_coords(x, y)
            try:
                pts = mk.get("points", [])
                if 0 <= int(vidx) < len(pts):
                    pts[int(vidx)] = (
                        max(0, min(self.img_w, ix)),
                        max(0, min(self.img_h, iy))
                    )
                    mk["points"] = pts
                    self._render_image_on_canvas()
            except Exception:
                pass
            return
        if mode == "draw":
            sx, sy = self._new_rect_start
            if self._new_rect_id:
                self.canvas.coords(self._new_rect_id, sx, sy, x, y)
        elif mode == "move":
            bbox = self.drag_data.get("bbox")
            sx, sy = self.drag_data.get("start")
            if not bbox:
                return
            dx = (x - sx) / max(1e-9, self.display_scale)
            dy = (y - sy) / max(1e-9, self.display_scale)
            self._apply_delta_to_bbox(bbox, dx, dy)
            self._clamp_bbox_to_image(bbox, self.img_w, self.img_h)
            self.drag_data["start"] = (x, y)
            self._render_image_on_canvas()
        elif mode == "resize":
            bbox = self.drag_data.get("bbox")
            handle_idx = self.drag_data.get("handle")
            if bbox is None:
                return
            ix, iy = self.canvas_to_image_coords(x, y)
            if handle_idx == 0:
                bbox.x1 = min(ix, bbox.x2 - 1)
                bbox.y1 = min(iy, bbox.y2 - 1)
            elif handle_idx == 1:
                bbox.x2 = max(ix, bbox.x1 + 1)
                bbox.y1 = min(iy, bbox.y2 - 1)
            elif handle_idx == 2:
                bbox.x2 = max(ix, bbox.x1 + 1)
                bbox.y2 = max(iy, bbox.y1 + 1)
            elif handle_idx == 3:
                bbox.x1 = min(ix, bbox.x2 - 1)
                bbox.y2 = max(iy, bbox.y1 + 1)
            self._render_image_on_canvas()
        elif mode == "zoomrect":
            sx, sy = self._zoom_rect_start
            if hasattr(self, "_zoom_rect_id") and self._zoom_rect_id:
                self.canvas.coords(self._zoom_rect_id, sx, sy, x, y)

    # ---------- BBOX  ----------
    def _apply_delta_to_bbox(self, bbox, dx: float, dy: float):
        """Translate bbox by dx, dy in image coordinates."""
        try:
            bbox.x1 += dx
            bbox.y1 += dy
            bbox.x2 += dx
            bbox.y2 += dy
        except Exception:
            pass

    def _clamp_bbox_to_image(self, bbox, W: int, H: int):
        """Clamp bbox to image bounds while preserving size."""
        try:
            w = bbox.x2 - bbox.x1
            h = bbox.y2 - bbox.y1
            if bbox.x1 < 0:
                bbox.x1 = 0; bbox.x2 = w
            if bbox.y1 < 0:
                bbox.y1 = 0; bbox.y2 = h
            if bbox.x2 > W:
                bbox.x2 = W; bbox.x1 = W - w
            if bbox.y2 > H:
                bbox.y2 = H; bbox.y1 = H - h
        except Exception:
            pass

    def _draw_handles(self, bbox: BBox):
        # draw small squares at the corners on top of canvas
        x1 = bbox.x1 * self.display_scale + self.offset_x
        y1 = bbox.y1 * self.display_scale + self.offset_y
        x2 = bbox.x2 * self.display_scale + self.offset_x
        y2 = bbox.y2 * self.display_scale + self.offset_y
        s = self.HANDLE_SIZE
        coords = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        for i, (cx, cy) in enumerate(coords):
            hid = self.canvas.create_rectangle(cx - s, cy - s, cx + s, cy + s, fill="#FFFFFF", outline="#000000", tags=("handle", f"bbox_{id(bbox)}", str(i)))
            self.handle_id_to_info[hid] = (bbox, i)

    def _highlight_selected_bbox(self, bbox: BBox):
        """Set and visually emphasize the selected bbox."""
        try:
            self.selected_bbox = bbox
            self._render_image_on_canvas()
        except Exception:
            # Fail-safe: ignore highlight errors
            pass

    def add_bbox_extra(self, x1, y1, x2, y2, label="circle"):
        """
        Add a bounding box to the 'extra' layer for the current image.
        Used when converting detections (e.g., circles ? boxes).
        """
        if not self.current_image_path:
            print("[WARN] No image loaded; cannot add extra bbox.")
            return

        img_path = self.current_image_path
        if not hasattr(self, "bboxes_extra"):
            self.bboxes_extra = {}
        if img_path not in self.bboxes_extra:
            self.bboxes_extra[img_path] = []

        bbox = {
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "label": label
        }

        # Store per image
        self.bboxes_extra[img_path].append(bbox)

        # ? Also keep a flat list for live shape count / previews
        if not hasattr(self, "mv_bboxes"):
            self.mv_bboxes = []
        self.mv_bboxes.append(bbox)

        # print(f"[DEBUG] Added extra bbox for '{label}': {bbox}")
        # print(f"[DEBUG] Total mv_bboxes now: {len(self.mv_bboxes)}")

        self._render_image_on_canvas()

    def find_bbox_at(self, x, y, tolerance=5):
        """
        Return the first *visible* bbox under the cursor.
        Respects visibility checkboxes (GT / Pred / Extra).
        """
        if not self.image_files or self.current_index < 0:
            return None, None  # (bbox, layer)

        ix, iy = self.canvas_to_image_coords(x, y)
        img_path = self.image_files[self.current_index]

        visible_layers = []
        if self.show_gt.get():
            visible_layers.append(("gt", self.bboxes_gt.get(img_path, [])))
        if self.show_pred.get():
            visible_layers.append(("pred", self.bboxes_pred.get(img_path, [])))
        if self.show_extra.get():
            visible_layers.append(("extra", self.bboxes_extra.get(img_path, [])))

        # Reverse so latest drawn boxes (top-most) are tested first
        for layer_name, boxes in reversed(visible_layers):
            for bbox in reversed(boxes):
                if bbox.x1 - tolerance <= ix <= bbox.x2 + tolerance and bbox.y1 - tolerance <= iy <= bbox.y2 + tolerance:
                    return bbox, layer_name

        return None, None

    def _point_in_polygon(self, px, py, points):
        """Ray casting point-in-polygon test in image coordinates."""
        inside = False
        n = len(points)
        if n < 3:
            return False
        xints = 0.0
        p1x, p1y = points[0]
        for i in range(n + 1):
            p2x, p2y = points[i % n]
            if py > min(p1y, p2y):
                if py <= max(p1y, p2y):
                    if p1y != p2y:
                        xints = (py - p1y) * (p2x - p1x) / (p2y - p1y + 1e-12) + p1x
                    if p1x == p2x or px <= max(p1x, p2x):
                        if px <= xints:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def find_seg_mask_at(self, x, y, tolerance=3):
        """Return (mask, layer) for first visible segmentation mask at canvas point (x,y)."""
        if not self.image_files or self.current_index < 0:
            return None, None
        ix, iy = self.canvas_to_image_coords(x, y)
        img_path = self.image_files[self.current_index]

        visible_layers = []
        if getattr(self, 'seg_show_gt', tk.BooleanVar(value=True)).get():
            visible_layers.append(("gt", self.seg_masks_gt.get(img_path, [])))
        if getattr(self, 'seg_show_pred', tk.BooleanVar(value=False)).get():
            visible_layers.append(("pred", self.seg_masks_pred.get(img_path, [])))
        if getattr(self, 'seg_show_extra', tk.BooleanVar(value=False)).get():
            visible_layers.append(("extra", self.seg_masks_extra.get(img_path, [])))

        for layer_name, masks in reversed(visible_layers):
            for mk in reversed(masks):
                pts = mk.get("points", [])
                if len(pts) < 3:
                    continue
                # quick bbox check
                xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
                minx, maxx = min(xs) - tolerance, max(xs) + tolerance
                miny, maxy = min(ys) - tolerance, max(ys) + tolerance
                if ix < minx or ix > maxx or iy < miny or iy > maxy:
                    continue
                if self._point_in_polygon(ix, iy, pts):
                    return mk, layer_name
        return None, None

    def find_seg_vertex_at(self, x, y, tolerance=6):
        """Return (mask, layer, vertex_index) if the canvas point (x,y) is near a mask vertex.
        Searches visible masks, preferring the currently selected mask if available.
        """
        if not self.image_files or self.current_index < 0:
            return None, None, None
        img_path = self.image_files[self.current_index]

        # Helper to iterate masks by layer respecting visibility
        def iter_visible_masks():
            layers = []
            if getattr(self, 'seg_show_gt', tk.BooleanVar(value=True)).get():
                layers.append(("gt", self.seg_masks_gt.get(img_path, [])))
            if getattr(self, 'seg_show_pred', tk.BooleanVar(value=False)).get():
                layers.append(("pred", self.seg_masks_pred.get(img_path, [])))
            if getattr(self, 'seg_show_extra', tk.BooleanVar(value=False)).get():
                layers.append(("extra", self.seg_masks_extra.get(img_path, [])))
            return layers

        # Prefer currently selected mask if any
        preferred = []
        if getattr(self, 'selected_seg_mask', None):
            # Find its layer by scanning
            for layer_name, masks in iter_visible_masks():
                for mk in masks:
                    if mk is self.selected_seg_mask:
                        preferred.append((layer_name, [mk]))
                        break
                if preferred:
                    break

        # Build full search order: selected first (if present), then all
        search_layers = preferred + list(iter_visible_masks())

        # Hit test against each vertex in canvas space
        for layer_name, masks in search_layers:
            for mk in masks or []:
                pts = mk.get("points", [])
                for i, (ix, iy) in enumerate(pts):
                    cx = int(ix * self.display_scale + self.offset_x)
                    cy = int(iy * self.display_scale + self.offset_y)
                    if abs(cx - x) <= tolerance and abs(cy - y) <= tolerance:
                        return mk, layer_name, i
        return None, None, None

    # ---------- Zoom helpers ----------
    def compute_base_scale(self, c_w: int, c_h: int) -> float:
        """Compute base scale for current image and canvas size."""
        if not self.img_w or not self.img_h:
            return 1.0
        if self.check_keep_ratio.get():
            return min(c_w / max(1, self.img_w), c_h / max(1, self.img_h))
        return c_w / max(1, self.img_w)

    def go_to_image(self, idx: int):
        """Clamp index and show image if changed."""
        if not self.image_files:
            return
        new_idx = max(0, min(int(idx), len(self.image_files) - 1))
        if new_idx != self.current_index:
            self.current_index = new_idx
            self.show_image_at_index()

    def fit_to_window(self):
        """Reset zoom and center the image to fit the canvas."""
        if not self.current_image:
            return

        c_w = self.canvas.winfo_width() or 1
        c_h = self.canvas.winfo_height() or 1
        base_scale = self.compute_base_scale(c_w, c_h)

        # Reset scale and offsets
        self.zoom_factor = 1.0
        self.display_scale = base_scale

        # Center image in the canvas
        disp_w = self.img_w * base_scale
        disp_h = self.img_h * base_scale
        self.offset_x = (c_w - disp_w) / 2
        self.offset_y = (c_h - disp_h) / 2

        # Make sure render doesn’t override this offset
        self._preserve_offset = True

        self._render_image_on_canvas()

    def zoom_at_cursor(self, cx, cy, factor):
        c_w = self.canvas.winfo_width() or 1
        c_h = self.canvas.winfo_height() or 1
        base_scale = self.compute_base_scale(c_w, c_h)
        # Clamp cursor to visible image area
        disp_w = int(self.img_w * self.display_scale) if getattr(self, 'display_scale', None) else int(self.img_w * base_scale * getattr(self, 'zoom_factor', 1.0))
        disp_h = int(self.img_h * self.display_scale) if getattr(self, 'display_scale', None) else int(self.img_h * base_scale * getattr(self, 'zoom_factor', 1.0))
        img_left = getattr(self, 'offset_x', (c_w - disp_w) / 2.0)
        img_top = getattr(self, 'offset_y', (c_h - disp_h) / 2.0)
        img_right = img_left + disp_w
        img_bottom = img_top + disp_h
        cx = max(img_left, min(img_right, cx))
        cy = max(img_top, min(img_bottom, cy))
        # image coords under cursor before zoom
        ix, iy = self.canvas_to_image_coords(cx, cy)
        # Store debug anchor for rendering
        self._debug_zoom_anchor = (cx, cy, ix, iy)
        self.zoom_factor *= factor
        new_display_scale = base_scale * self.zoom_factor
        # compute new offsets so the image point (ix,iy) remains under the cursor
        self.offset_x = cx - ix * new_display_scale
        self.offset_y = cy - iy * new_display_scale
        # preserve offsets so _render_image_on_canvas won't recenter
        self._preserve_offset = True
        self._render_image_on_canvas()

    def pan(self, dx_dir, dy_dir, step_px: int = None):
        """Pan the displayed image by a directional step.

        dx_dir/dy_dir should be -1, 0 or 1 indicating direction. step_px if provided is pixel step
        in canvas coords; otherwise computed as 20% of the smaller canvas dimension.
        """
        # compute step
        c_w = self.canvas.winfo_width() or 1
        c_h = self.canvas.winfo_height() or 1
        if step_px is None:
            step = max(50, int(0.2 * min(c_w, c_h)))
        else:
            step = int(step_px)
        # move offsets (note: offsets are in canvas pixels)
        self.offset_x += dx_dir * step
        self.offset_y += dy_dir * step
        # preserve offset so we don't recenter
        self._preserve_offset = True
        self._render_image_on_canvas()

    def pan_pixels(self, dx: int, dy: int):
        """Pan the displayed image by raw pixel deltas (canvas coordinates)."""
        self.offset_x += int(dx)
        self.offset_y += int(dy)
        self._preserve_offset = True
        self._render_image_on_canvas()

    def on_right_click(self, event):
        # Finalize segmentation polygon on right-click when drawing
        if getattr(self, '_active_left_tab', tk.StringVar(value='bbox')).get() == 'seg' and getattr(self, '_seg_is_drawing', False):
            try:
                self._seg_finalize_polygon()
            except Exception:
                pass
            return
        # Delegate hit-testing to a single helper
        x, y = event.x, event.y
        bbox, _layer = self.find_bbox_at(x, y)
        if bbox:
            self.selected_bbox = bbox
            self._render_image_on_canvas()

    # ---------- Segmentation helpers ----------
    def _seg_on_click_add_point(self, cx, cy):
        ix, iy = self.canvas_to_image_coords(cx, cy)
        if not self._seg_is_drawing:
            self._seg_is_drawing = True
            self._seg_points = [(ix, iy)]
            # create preview polygon
            try:
                self._seg_preview_poly_id = self.canvas.create_polygon(
                    cx, cy, outline="#00CCCC", fill="", width=2, dash=(3, 3), tags=("seg_preview",)
                )
            except Exception:
                self._seg_preview_poly_id = None
        else:
            self._seg_points.append((ix, iy))
        self._seg_update_preview_polygon()

    def _seg_update_preview_polygon(self):
        if not self._seg_is_drawing:
            return
        pts = []
        for (ix, iy) in self._seg_points:
            cx = int(ix * self.display_scale + self.offset_x)
            cy = int(iy * self.display_scale + self.offset_y)
            pts.extend([cx, cy])
        try:
            if self._seg_preview_poly_id is None:
                self._seg_preview_poly_id = self.canvas.create_polygon(
                    *pts, outline="#00CCCC", fill="", width=2, dash=(3, 3), tags=("seg_preview",)
                )
            else:
                if pts:
                    self.canvas.coords(self._seg_preview_poly_id, *pts)
        except Exception:
            pass

    def _seg_update_preview_edge(self, cx, cy):
        if not self._seg_is_drawing or not self._seg_points:
            return
        last_ix, last_iy = self._seg_points[-1]
        x1 = int(last_ix * self.display_scale + self.offset_x)
        y1 = int(last_iy * self.display_scale + self.offset_y)
        try:
            if self._seg_preview_line_id is None:
                self._seg_preview_line_id = self.canvas.create_line(
                    x1, y1, cx, cy, fill="#00CCCC", dash=(3, 3), width=2, tags=("seg_preview_edge",)
                )
            else:
                self.canvas.coords(self._seg_preview_line_id, x1, y1, cx, cy)
        except Exception:
            pass

    def _seg_finalize_polygon(self):
        if len(self._seg_points) < 3:
            self._seg_clear_preview()
            return
        try:
            img_path = self.image_files[self.current_index]
        except Exception:
            self._seg_clear_preview()
            return
        try:
            cls = int(self.parse_selected_seg_class())
        except Exception:
            cls = 0
        mask = {"cls": cls, "points": list(self._seg_points)}
        self.seg_masks_gt.setdefault(img_path, []).append(mask)
        try:
            self.update_dataset_info()
        except Exception:
            pass
        self._seg_clear_preview()
        self._render_image_on_canvas()

    def _seg_clear_preview(self):
        self._seg_is_drawing = False
        self._seg_points = []
        try:
            if self._seg_preview_poly_id:
                self.canvas.delete(self._seg_preview_poly_id)
        except Exception:
            pass
        try:
            if self._seg_preview_line_id:
                self.canvas.delete(self._seg_preview_line_id)
        except Exception:
            pass
        self._seg_preview_poly_id = None
        self._seg_preview_line_id = None

    def delete_all_seg_gt_for_current(self):
        try:
            if not self.image_files:
                return
            img_path = self.image_files[self.current_index]
            if not self.seg_masks_gt.get(img_path):
                messagebox.showinfo("No GT Masks", "No GT segmentation masks to delete for current image.")
                return
            if not messagebox.askyesno("Confirm", "Delete all GT segmentation masks for current image?"):
                return
            self.seg_masks_gt[img_path] = []
            # optionally remove seg file on disk
            if self.label_folder:
                fname = os.path.splitext(os.path.basename(img_path))[0]
                seg_txt_path = os.path.join(self.label_folder, fname + "_seg.txt")
                try:
                    if os.path.exists(seg_txt_path):
                        os.remove(seg_txt_path)
                except Exception:
                    pass
            self._render_image_on_canvas()
        except Exception:
            pass

    def seg_validate_all_extra_to_gt(self):
        try:
            if not self.image_files:
                return
            img_path = self.image_files[self.current_index]
            extras = list(self.seg_masks_extra.get(img_path, []))
            if not extras:
                messagebox.showinfo("Nothing to Validate", "No extra segmentation masks to validate.")
                return
            self.seg_masks_gt.setdefault(img_path, []).extend(extras)
            self.seg_masks_extra[img_path] = []
            self._render_image_on_canvas()
            messagebox.showinfo("Validated", f"Moved {len(extras)} extra mask(s) to GT.")
        except Exception:
            pass

    def save_current_seg_masks(self):
        """Save GT segmentation masks for the current image in YOLO-seg polygon format.
        One line per instance: cls x1 y1 x2 y2 ... with normalized coords in [0,1].
        Saved as <label_folder>/<image_name>_seg.txt to avoid clobbering bbox labels.
        """
        try:
            if not self.label_folder:
                messagebox.showwarning("No Labels Folder", "Select a folder for labels first.")
                return
            if not self.image_files:
                return
            img_path = self.image_files[self.current_index]
            fname = os.path.splitext(os.path.basename(img_path))[0]
            txt_path = os.path.join(self.label_folder, fname + "_seg.txt")
            masks = self.seg_masks_gt.get(img_path, [])
            if not masks:
                messagebox.showinfo("No Masks", "No GT masks to save for this image.")
                return
            lines = []
            W = max(1, int(self.img_w))
            H = max(1, int(self.img_h))
            for mk in masks:
                cls = int(mk.get("cls", 0))
                pts = mk.get("points", [])
                if len(pts) < 3:
                    continue
                coords = []
                for (x, y) in pts:
                    nx = min(1.0, max(0.0, float(x) / W))
                    ny = min(1.0, max(0.0, float(y) / H))
                    coords.append(f"{nx:.6f} {ny:.6f}")
                if not coords:
                    continue
                lines.append(f"{cls} " + " ".join(coords))
            with open(txt_path, "w") as f:
                f.write("\n".join(lines))
            messagebox.showinfo("Saved", f"Segmentation saved to:\n{txt_path}")
        except Exception as e:
            try:
                messagebox.showerror("Save Error", str(e))
            except Exception:
                print("[ERROR] save_current_seg_masks:", e)

    # ---------- Validation helpers ----------
    def validate_all_predictions(self):
        """Move all predictions to GT for current image."""
        if not self.image_files:
            return
        img_path = self.image_files[self.current_index]
        preds = list(self.bboxes_pred.get(img_path, []))
        if not preds:
            messagebox.showinfo("No Predictions", "No predictions to validate for this image.")
            return
        self.bboxes_gt.setdefault(img_path, []).extend(preds)
        self.bboxes_pred[img_path] = []
        self._render_image_on_canvas()
        messagebox.showinfo("Validated", f"Moved {len(preds)} predictions to GT.")

    def validate_all_extra_to_gt(self):
        """Promote extra predicted boxes (unmatched preds) into GT."""
        if not self.image_files:
            return
        img_path = self.image_files[self.current_index]
        preds = list(self.bboxes_pred.get(img_path, []))
        gts = self.bboxes_gt.get(img_path, [])
        extra_preds = []
        for p in preds:
            matched = False
            for g in gts:
                if calculate_iou(p.as_tuple(), g.as_tuple()) > 0.5:
                    matched = True; break
            if not matched:
                extra_preds.append(p)
        if not extra_preds:
            messagebox.showinfo("Nothing to Validate", "No extra predictions to validate.")
            return
        for p in extra_preds:
            if p in self.bboxes_pred.get(img_path, []):
                try: self.bboxes_pred[img_path].remove(p)
                except: pass
            self.bboxes_gt.setdefault(img_path, []).append(p)
        if self.selected_bbox in extra_preds:
            self.selected_bbox = None
        self._render_image_on_canvas()
        messagebox.showinfo("Validated", f"Moved {len(extra_preds)} extra prediction(s) to GT.")

    def validate_selected_prediction(self):
        if not self.selected_bbox:
            messagebox.showwarning("No Selection", "No bbox selected.")
            return
        img_path = self.image_files[self.current_index]
        if self.selected_bbox not in self.bboxes_pred.get(img_path, []):
            messagebox.showwarning("No Selection", "Selected bbox is not a prediction.")
            return
        self.bboxes_pred[img_path].remove(self.selected_bbox)
        self.bboxes_gt.setdefault(img_path, []).append(self.selected_bbox)
        self.selected_bbox = None
        self._render_image_on_canvas()

    # ---------- Save / delete ----------
    def save_current_annotations(self):
        if not self.label_folder:
            messagebox.showwarning("No Labels Folder", "Select a folder for labels first.")
            return
        if not self.image_files:
            return
        img_path = self.image_files[self.current_index]
        fname = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(self.label_folder, fname + ".txt")
        bboxes = self.bboxes_gt.get(img_path, [])
        lines = []
        for bb in bboxes:
            nx, ny, nw, nh = bb.normalize(self.img_w, self.img_h)
            nx = min(1.0, max(0.0, nx))
            ny = min(1.0, max(0.0, ny))
            nw = min(1.0, max(0.0, nw))
            nh = min(1.0, max(0.0, nh))
            lines.append(f"{bb.cls} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")
        try:
            with open(txt_path, "w") as f:
                f.write("\n".join(lines))
            # suppress popup printing if user prefers (we won't print)
            # show a small info
            messagebox.showinfo("Saved", f"Annotations saved:\n{txt_path}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def delete_current_image_and_label(self):
        if not self.image_files or self.current_index < 0:
            return
        img_path = self.image_files[self.current_index]
        fname = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = os.path.join(self.label_folder, fname + ".txt")
        pred_txt_path = os.path.join(self.prediction_folder, fname + ".txt")
        try:
            os.remove(img_path)
            if os.path.exists(txt_path):
                os.remove(txt_path)
            if os.path.exists(pred_txt_path):
                os.remove(pred_txt_path)
            self.load_image_list()
            # reload annotations for remaining images (load_image_list resets bbox dicts)
            try:
                self.load_annotations_for_all_images()
            except Exception:
                pass
            self.current_index = min(self.current_index, len(self.image_files) - 1)
            self.show_image_at_index()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete: {e}")

    def delete_all_gt_for_current(self):
        if not self.image_files:
            return
        img_path = self.image_files[self.current_index]
        if not self.bboxes_gt.get(img_path):
            messagebox.showinfo("No GT", "No GT annotations to delete for current image.")
            return
        if not messagebox.askyesno("Confirm", "Delete all GT annotations for current image?"):
            return
        self.bboxes_gt[img_path] = []
        # optionally remove label file on disk:
        if self.label_folder:
            fname = os.path.splitext(os.path.basename(img_path))[0]
            txt_path = os.path.join(self.label_folder, fname + ".txt")
            try:
                if os.path.exists(txt_path):
                    os.remove(txt_path)
            except Exception:
                pass
        self._render_image_on_canvas()

    # ---------- Navigation ----------
    def next_image(self):
        if not self.image_files:
            return
        self.go_to_image(self.current_index + 1)

    def prev_image(self):
        if not self.image_files:
            return
        self.go_to_image(self.current_index - 1)

    def _on_slider_move(self, value):
        if not self.image_files:
            return
        try:
            self.go_to_image(int(float(value)))
        except Exception:
            pass
    
    def _register_mode_button(self, mode_key, button):
        self.mode_buttons.setdefault(mode_key, []).append(button)

    def _darken_color(self, hex_color, factor=0.8):
        """Return a darker shade of the provided hex color."""
        try:
            hex_color = hex_color.lstrip("#")
            if len(hex_color) != 6:
                return f"#{hex_color}"
            r = int(hex_color[0:2], 16)
            g = int(hex_color[2:4], 16)
            b = int(hex_color[4:6], 16)
            r = max(0, min(255, int(r * factor)))
            g = max(0, min(255, int(g * factor)))
            b = max(0, min(255, int(b * factor)))
            return f"#{r:02x}{g:02x}{b:02x}"
        except Exception:
            return "#444444"

    def update_mode_buttons_look(self):
        for mk, buttons in self.mode_buttons.items():
            cfg = MODES.get(mk) or SEGMENTATION_ONLY_MODES.get(mk, {})
            base = cfg.get("color", "#DDDDDD")
            active = self._darken_color(base)
            selected = self.current_mode.get() == mk
            for btn in buttons:
                btn.configure(bg=active if selected else base,
                              relief=tk.SUNKEN if selected else tk.RAISED)

    def set_mode(self, mode_key):
        if self.current_mode.get() == mode_key:
            self.current_mode.set("none")
        else:
            self.current_mode.set(mode_key)
        self.update_mode_buttons_look()

    def _mv_get_bboxes_from_shape(self, shape_name):
        """
        Convert any detected shape into bounding boxes [(x1, y1, x2, y2)].

        Supports:
            - circles      -> (x, y, r)
            - circularity  -> (x, y, r)
            - rects        -> (x, y, w, h)
            - triangles    -> [(x, y), ...]
            - polygons     -> [(x, y), ...]
            - direct bboxes -> (x1, y1, x2, y2)

        Also respects optional tuning parameters:
            - hough_min_r / hough_max_r (for circle & circularity)
            - Image bounds for clipping
        """
        import numpy as np

        shape_name = (shape_name or "").lower()
        bboxes = []

        # --- Map shape names to detected attributes ---
        shape_attr_map = {
            "circle": "detected_circles",
            "circularity": "detected_circularity",
            "rect": "detected_rects",
            "square": "detected_rects",
            "triangle": "detected_triangles",
            "polygon": "detected_polygons",
            "contour": "detected_contours",
        }

        # --- Determine the matching attribute name ---
        attr_name = None
        for key, val in shape_attr_map.items():
            if key in shape_name:
                attr_name = val
                break

        if not attr_name or not hasattr(self, attr_name):
            print(f"[WARN] No detected data for shape: {shape_name}")
            return []

        detected = getattr(self, attr_name, [])
        if not detected:
            print(f"[WARN] No detections stored in {attr_name}")
            return []

        # --- Retrieve optional radius tuning parameters from sliders ---
        min_r = int(self.hough_min_r.get()) if hasattr(self, "hough_min_r") else 0
        max_r = int(self.hough_max_r.get()) if hasattr(self, "hough_max_r") else 9999

        # --- Get image bounds for clipping ---
        if getattr(self, "current_image", None):
            W, H = self.current_image.size
        else:
            W = H = None

        def _clip(v, lo, hi):
            return max(lo, min(hi, v))

        # --- Conversion loop ---
        for shape in detected:
            try:
                # 1?? Already a bbox (x1, y1, x2, y2)
                if (
                    isinstance(shape, (list, tuple))
                    and len(shape) == 4
                    and all(isinstance(v, (int, float)) for v in shape)
                ):
                    x1, y1, x2, y2 = map(int, shape)
                    bboxes.append((x1, y1, x2, y2))

                # 2?? Circle or circularity (x, y, r)
                elif (
                    isinstance(shape, (list, tuple))
                    and len(shape) == 3
                    and all(isinstance(v, (int, float)) for v in shape)
                ):
                    x, y, r = map(int, shape)
                    if r < min_r or r > max_r:
                        continue  # respect slider tuning
                    x1, y1, x2, y2 = x - r, y - r, x + r, y + r
                    if W and H:
                        x1 = _clip(x1, 0, W - 1)
                        y1 = _clip(y1, 0, H - 1)
                        x2 = _clip(x2, 0, W - 1)
                        y2 = _clip(y2, 0, H - 1)
                    bboxes.append((x1, y1, x2, y2))

                # 3?? Polygon / triangle: list of (x, y)
                elif (
                    isinstance(shape, (list, tuple))
                    and all(isinstance(p, (list, tuple)) and len(p) == 2 for p in shape)
                ):
                    arr = np.array(shape)
                    x1, y1 = np.min(arr[:, 0]), np.min(arr[:, 1])
                    x2, y2 = np.max(arr[:, 0]), np.max(arr[:, 1])
                    if W and H:
                        x1 = _clip(int(x1), 0, W - 1)
                        y1 = _clip(int(y1), 0, H - 1)
                        x2 = _clip(int(x2), 0, W - 1)
                        y2 = _clip(int(y2), 0, H - 1)
                    bboxes.append((x1, y1, x2, y2))

                else:
                    print(f"[WARN] Unknown shape format: {shape}")

            except Exception as e:
                print(f"[ERROR] Could not convert {shape}: {e}")

        print(f"[INFO] Extracted {len(bboxes)} bounding boxes from {shape_name}")
        return bboxes

    def _mv_add_shape_to_extras(self, shape_name):
        """
        Add detected shapes' bounding boxes to the Extras layer (self.bboxes_extra)
        so they appear in the left panel with tickboxes/sliders.
        """
        from core.models import BBox
        import os, re

        bboxes = self._mv_get_bboxes_from_shape(shape_name)
        if not bboxes:
            print(f"[WARN] No bounding boxes found for {shape_name}")
            return

        # Get current image key (path or name)
        img_path = getattr(self, "current_image_path", None)
        if not img_path:
            print("[ERROR] No current image path found. Cannot assign Extras.")
            return

        # Ensure the dictionary exists and has an entry for this image
        if not hasattr(self, "bboxes_extra"):
            self.bboxes_extra = {}
        if img_path not in self.bboxes_extra:
            self.bboxes_extra[img_path] = []

        # Map shape name to class index (for YOLO compatibility)
        if not hasattr(self, "class_index_map"):
            self.class_index_map = {}

        norm_name = re.sub(r"[^a-z]", "", shape_name.lower())
        if norm_name not in self.class_index_map:
            self.class_index_map[norm_name] = len(self.class_index_map)

        cls = self.class_index_map[norm_name]

        added = 0
        for (x1, y1, x2, y2) in bboxes:
            try:
                box = BBox(cls, x1, y1, x2, y2)
                self.bboxes_extra[img_path].append(box)
                added += 1
            except Exception as e:
                print(f"[ERROR] Failed to create BBox: {e}")

        print(f"[INFO] Added {added} {shape_name} boxes to Extras for image {os.path.basename(img_path)}")

        # Refresh the Extras panel (left UI list)
        if hasattr(self, "refresh_extras_panel"):
            self.refresh_extras_panel()
        else:
            self._render_image_on_canvas()


# ---------- Left Panel Builders ----------
    def _make_left_slider_row(self, parent, label, var_show, var_transp, color, style_name):
        from tkinter import ttk
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=0)
        ttk.Checkbutton(row, text=label, variable=var_show,
                        command=self._render_image_on_canvas).pack(side=tk.LEFT, padx=(0, 6))
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(f"{style_name}.Horizontal.TScale", troughcolor="#f0f0f0")
        style.map(f"{style_name}.Horizontal.TScale",
                  background=[("active", color), ("!active", color)])
        ttk.Scale(row, from_=0, to=1, orient="horizontal", variable=var_transp,
                  length=90, style=f"{style_name}.Horizontal.TScale",
                  command=lambda e: self._render_image_on_canvas()).pack(side=tk.RIGHT, padx=2, pady=2)

    def _build_left_folder_section(self, parent):
        from tkinter import ttk, font
        bold_font = font.Font(family="TkDefaultFont", size=10, weight="bold")
        ttk.Label(parent, text="Folder", font=bold_font).pack(anchor=tk.W, pady=(6, 0))
        ttk.Button(parent, text="Select Parent Folder", command=self.select_parent_folder).pack(fill=tk.X, pady=2)
        ttk.Separator(parent).pack(fill=tk.X, pady=6)

    def _build_left_tabs(self, parent):
        from tkinter import ttk
        left_tabs = ttk.Notebook(parent)
        tab_bbox = ttk.Frame(left_tabs)
        tab_seg = ttk.Frame(left_tabs)
        left_tabs.add(tab_bbox, text="BBox")
        left_tabs.add(tab_seg, text="Segmentation")
        left_tabs.pack(fill=tk.BOTH, expand=True)
        return left_tabs, tab_bbox, tab_seg

    def _build_left_bbox_tab(self, tab_bbox):
        import tkinter as tk
        from tkinter import ttk, font
        bold_font = font.Font(family="TkDefaultFont", size=10, weight="bold")

        ttk.Label(tab_bbox, text="Visibility & Transparency", font=bold_font).pack(anchor=tk.W, pady=(6, 2))
        self._make_left_slider_row(tab_bbox, "GT", self.show_gt, self.transparency_gt, "#3cb043", "Green")
        self._make_left_slider_row(tab_bbox, "Pred", self.show_pred, self.transparency_pred, "#ff5050", "Red")
        self._make_left_slider_row(tab_bbox, "Extra", self.show_extra, self.transparency_extra, "#0090ff", "Blue")

        ttk.Separator(tab_bbox).pack(fill=tk.X, pady=6)

        ttk.Label(tab_bbox, text="Select Class", font=bold_font).pack(anchor=tk.W, pady=(4, 2))
        self.cls_var = tk.StringVar(value="No classes loaded")
        self.class_selector = ttk.Combobox(
            tab_bbox,
            textvariable=self.cls_var,
            values=["No classes loaded"],
            state="disabled",
            width=20
        )
        self.class_selector.pack(anchor=tk.W, pady=(0, 4))
        # sync seg selector when bbox selector changes
        def _on_bbox_class_change(event=None):
            try:
                val = self.class_selector.get()
                if hasattr(self, 'seg_class_selector') and val:
                    self.seg_class_selector.set(val)
            except Exception:
                pass
        try:
            self.class_selector.bind('<<ComboboxSelected>>', _on_bbox_class_change)
        except Exception:
            pass

        ttk.Label(tab_bbox, text="Modes", font=bold_font).pack(anchor=tk.W)
        modes_frame = ttk.Frame(tab_bbox)
        modes_frame.pack(fill=tk.X, pady=1)
        for mode_key, cfg in MODES.items():
            if mode_key == "none":
                continue
            b = tk.Button(
                modes_frame, text=cfg["label"],
                relief=tk.RAISED, bg=cfg["color"], fg="#000000",
                command=lambda k=mode_key: self.set_mode(k)
            )
            b.pack(fill=tk.X, pady=1)
            self._register_mode_button(mode_key, b)

        ttk.Separator(tab_bbox).pack(fill=tk.X, pady=6)

        style = ttk.Style()
        style.configure("Green.TButton", background="#12c412", foreground="white")
        style.configure("Red.TButton", background="#cc0a0a", foreground="white")

        ttk.Label(tab_bbox, text="Editing", font=bold_font).pack(anchor=tk.W)
        ttk.Button(tab_bbox, text="Delete All GT", style="Red.TButton",
                   command=self.delete_all_gt_for_current).pack(fill=tk.X, pady=2)
        ttk.Button(tab_bbox, text="Delete Image & Label", style="Red.TButton",
                   command=self.delete_current_image_and_label).pack(fill=tk.X, pady=2)
        ttk.Button(tab_bbox, text="Validate All Extra -> GT", style="Green.TButton",
                   command=self.validate_all_extra_to_gt).pack(fill=tk.X, pady=2)
        ttk.Button(tab_bbox, text="Save GT (overwrite .txt)", style="Green.TButton",
                   command=self.save_current_annotations).pack(fill=tk.X, pady=2)

        ttk.Separator(tab_bbox).pack(fill=tk.X, pady=6)
        ttk.Label(tab_bbox, text="Navigation", font=bold_font).pack(anchor=tk.W)
        ttk.Button(tab_bbox, text="Fit to Window", command=self.fit_to_window).pack(fill=tk.X, pady=0)

        # Pan buttons
        pan_frame = ttk.Frame(tab_bbox)
        pan_frame.pack(pady=(6, 2), anchor="center")
        font_big = font.Font(size=12, weight="bold")
        tk.Button(pan_frame, text="↑", width=4, font=font_big, bg="#087b18", fg="white",
                command=lambda: self.pan(0, 1)).grid(row=1, column=1)
        tk.Button(pan_frame, text="←", width=4, font=font_big, bg="#087b18", fg="white",
                command=lambda: self.pan(1, 0)).grid(row=2, column=0)
        tk.Button(pan_frame, text="→", width=4, font=font_big, bg="#087b18", fg="white",
                command=lambda: self.pan(-1, 0)).grid(row=2, column=2)
        tk.Button(pan_frame, text="↓", width=4, font=font_big, bg="#087b18", fg="white",
                command=lambda: self.pan(0, -1)).grid(row=3, column=1)
        for i in range(3):
            pan_frame.columnconfigure(i, weight=1)
        for i in range(4):
            pan_frame.rowconfigure(i, weight=1)

        ttk.Separator(tab_bbox).pack(fill=tk.X, pady=6)

    def _build_left_seg_tab(self, tab_seg):
        import tkinter as tk
        from tkinter import ttk, font
        bold_font = font.Font(family="TkDefaultFont", size=10, weight="bold")

        ttk.Label(tab_seg, text="Visibility & Transparency", font=bold_font).pack(anchor=tk.W, pady=(6, 2))
        self._make_left_slider_row(tab_seg, "GT", self.seg_show_gt, self.seg_transparency_gt, "#3cb043", "SegGreen")
        self._make_left_slider_row(tab_seg, "Pred", self.seg_show_pred, self.seg_transparency_pred, "#ff5050", "SegRed")
        self._make_left_slider_row(tab_seg, "Extra", self.seg_show_extra, self.seg_transparency_extra, "#0090ff", "SegBlue")

        ttk.Separator(tab_seg).pack(fill=tk.X, pady=6)

        ttk.Label(tab_seg, text="Select Class", font=bold_font).pack(anchor=tk.W, pady=(4, 2))
        self.seg_class_selector = ttk.Combobox(
            tab_seg,
            textvariable=self.seg_cls_var,
            values=["0"],
            state="readonly",
            width=20
        )
        self.seg_class_selector.pack(anchor=tk.W, pady=(0, 4))
        # initial populate if dataset already known
        try:
            self.update_seg_class_selector()
        except Exception:
            pass
        # sync bbox selector when seg selector changes
        def _on_seg_class_change(event=None):
            try:
                val = self.seg_class_selector.get()
                if hasattr(self, 'class_selector') and val:
                    self.class_selector.set(val)
            except Exception:
                pass
        try:
            self.seg_class_selector.bind('<<ComboboxSelected>>', _on_seg_class_change)
        except Exception:
            pass

        ttk.Label(tab_seg, text="Modes", font=bold_font).pack(anchor=tk.W)
        modes_frame = ttk.Frame(tab_seg)
        modes_frame.pack(fill=tk.X, pady=1)
        for mode_key, cfg in MODES.items():
            if mode_key == "none":
                continue
            b = tk.Button(
                modes_frame, text=cfg["label"],
                relief=tk.RAISED, bg=cfg["color"], fg="#000000",
                command=lambda k=mode_key: self.set_mode(k)
            )
            b.pack(fill=tk.X, pady=1)
            self._register_mode_button(mode_key, b)
        for mode_key, cfg in SEGMENTATION_ONLY_MODES.items():
            b = tk.Button(
                modes_frame, text=cfg["label"],
                relief=tk.RAISED, bg=cfg["color"], fg="#000000",
                command=lambda k=mode_key: self.set_mode(k)
            )
            b.pack(fill=tk.X, pady=1)
            self._register_mode_button(mode_key, b)

        ttk.Separator(tab_seg).pack(fill=tk.X, pady=6)

        style = ttk.Style()
        style.configure("Green.TButton", background="#12c412", foreground="white")
        style.configure("Red.TButton", background="#cc0a0a", foreground="white")

        ttk.Label(tab_seg, text="Editing", font=bold_font).pack(anchor=tk.W)
        ttk.Button(tab_seg, text="Delete All GT Masks", style="Red.TButton",
                   command=self.delete_all_seg_gt_for_current).pack(fill=tk.X, pady=2)
        ttk.Button(tab_seg, text="Validate All Extra -> GT Masks", style="Green.TButton",
                   command=self.seg_validate_all_extra_to_gt).pack(fill=tk.X, pady=2)
        ttk.Button(tab_seg, text="Save GT Masks (YOLO-seg)", style="Green.TButton",
                   command=self.save_current_seg_masks).pack(fill=tk.X, pady=2)
        
        
        ttk.Separator(tab_seg).pack(fill=tk.X, pady=6)
        ttk.Label(tab_seg, text="Navigation", font=bold_font).pack(anchor=tk.W)
        ttk.Button(tab_seg, text="Fit to Window", command=self.fit_to_window).pack(fill=tk.X, pady=0)

        # Pan buttons
        pan_frame = ttk.Frame(tab_seg)
        pan_frame.pack(pady=(6, 2), anchor="center")
        font_big = font.Font(size=12, weight="bold")
        tk.Button(pan_frame, text="↑", width=4, font=font_big, bg="#087b18", fg="white",
                command=lambda: self.pan(0, 1)).grid(row=1, column=1)
        tk.Button(pan_frame, text="←", width=4, font=font_big, bg="#087b18", fg="white",
                command=lambda: self.pan(1, 0)).grid(row=2, column=0)
        tk.Button(pan_frame, text="→", width=4, font=font_big, bg="#087b18", fg="white",
                command=lambda: self.pan(-1, 0)).grid(row=2, column=2)
        tk.Button(pan_frame, text="↓", width=4, font=font_big, bg="#087b18", fg="white",
                command=lambda: self.pan(0, -1)).grid(row=3, column=1)
        for i in range(3):
            pan_frame.columnconfigure(i, weight=1)
        for i in range(4):
            pan_frame.rowconfigure(i, weight=1)

        ttk.Separator(tab_seg).pack(fill=tk.X, pady=6)



    def _build_left_panel(self):
        from tkinter import ttk
        left = ttk.Frame(self)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=6)
        self._build_left_folder_section(left)
        _tabs, tab_bbox, tab_seg = self._build_left_tabs(left)
        self.left_tabs = _tabs
        # track active tab name -> bbox/seg
        def _on_tab_changed(event=None):
            try:
                sel = self.left_tabs.tab(self.left_tabs.select(), 'text')
                self._active_left_tab.set('seg' if 'Segmentation' in sel else 'bbox')
            except Exception:
                pass
        self.left_tabs.bind('<<NotebookTabChanged>>', _on_tab_changed)
        _on_tab_changed()
        self._build_left_bbox_tab(tab_bbox)
        self._build_left_seg_tab(tab_seg)
        return left

# ---------- Run ----------
if __name__ == "__main__":
    app = YoloEditorApp()
    app.mainloop()



