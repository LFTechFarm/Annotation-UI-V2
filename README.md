# 📘 Annotation-UI-V2  
### A Lightweight, Powerful Annotation Tool for Bounding Boxes & Masks

**Annotation-UI-V2** is a Python/Tkinter graphical interface designed to **speed up dataset annotation** for computer vision tasks.  
It supports **bounding boxes**, **segmentation masks**, YOLO-based predictions, and classical Machine Vision tools for assisted labeling.

---

# 1️⃣ Getting Started — Full Tutorial

## 1.1 Launch the Application  
Start Annotation-UI-V2 by:
- Double-clicking the executable  
**or**  
- Running the Python script  

The interface will appear as shown:  
<img width="1915" height="1052" alt="image" src="https://github.com/user-attachments/assets/b0d71930-9995-4583-9570-5b201b01b6df" />


---

## 1.2 Load Your Dataset 📂  
Click **“Select Parent Folder”** (top-left).  
Your dataset must follow YOLO format:

dataset/<br>
├── images/ # your .jpg/.png files<br>
└── labels/ # your YOLO .txt annotation files<br>


If you only have images, create an empty `labels/` folder.

---

## 1.3 Select Your Annotation Task  
Using the tabs on the LEFT panel, choose the desired task:

- **Bounding Box (Detection)**  
- **Mask (Segmentation)**  

Your tools will update automatically based on your selection.

---

## 1.4 UI Layout Overview  
The interface is split into two operative areas:

### 🔵 Left Panel = **Task Tools (Your Annotation Actions)**  
Used for navigation, class selection, and annotation.

### 🟣 Right Panel = **Helpers (Dataset Info, YOLO, Machine Vision)**  
Used for automation, predictions, and shape detection.

This separation keeps annotation clean and efficient.

---

# 2️⃣ Left Panel — Task Tools

## 2.1 Class Selection (Required Before Annotating)  
Before drawing or editing anything, you must **select the class** of the object you want to annotate.

- Choose the target class using the class selector on the left  
- All drawings (BBox or Mask) will be tagged with this class  
- You can change the class later using **Change Class**  

This step is required for both **Bounding Box** and **Mask** annotations.

---

## 2.2 Image Navigation & Display  
- Move between images with the **slider** or **arrow buttons**  
- Zoom and pan to inspect details  
- Toggle visibility of:
  - Ground Truth (GT)
  - Predictions
  - Extra boxes  
- Adjust transparency with sliders  

---

## 2.3 Annotation Modes — Bounding Boxes  
Active when **BBox** task is selected.  
Class selection must be done before drawing.

| Mode | Description |
|------|-------------|
| ✏️ **Draw** | Create a new bounding box |
| ✂️ **Move / Resize** | Adjust the selected box |
| 🗑️ **Delete** | Remove a box |
| 🔀 **Change Class** | Modify the object class |
| ✨ **Validate** | Move predicted boxes to GT |

---

## 2.4 Annotation Modes — Masks  
Active when **Mask** task is selected.  
Class selection must be done before drawing.

| Mode | Description |
|------|-------------|
| ✏️ **Draw Mask** | Create a new segmentation mask (right click to validate) |
| ⚒️ **Edit Mask** | Move contour points to refine shape |
| 🗑️ **Delete Mask** | Remove the mask |
| 🔀 **Change Class** | Modify the object class |
| ✨ **Validate** | Transfer predicted mask to GT |
| ⚒️ **Modify Shape** | Modify the contour of the mask |

---

# 3️⃣ Right Panel — Helper Sections

## 3.1 Dataset Description  
Provides an overview of:

### **Dataset Overview**
- Total classes  
- Total images  
- Average boxes per image  
- Total annotations  
- Class balance  

### **Current Image**
- File name  
- Number of annotations (GT / PRED / EXTRA)  
- Date  

### **Class Legend**
Default class names are **class_{id}**.  
You can load a YAML file to display real class names.

Useful for monitoring progress and dataset integrity.

---

## 3.2 YOLO Prediction Helper 🤖  
Integrate YOLO to generate assisted annotations.

### Workflow:
1. Load YOLO weights (`best.pt`)  
2. Click **Predict Image**  
3. Toggle prediction visibility  
4. Validate predictions:
   - Individually  
   - Or all at once  
5. Use **Extra** (IOU Matching) to detect missing boxes  

Efficient for semi-automatic dataset annotation.

---

## 3.3 Machine Vision Tools 📷  
Classical CV algorithms to automatically detect geometric shapes or generate color/vegetation indices.

### Workflow:
1. Select image index (predefined or custom)
2. Generate a **binary mask** using thresholding  
3. Apply shape detection (tick MV preview):
   - ⚪ Circle  
   - 🔺 Triangle  
   - ⬛ Rectangle  
   - 🔷 Polygon  
   - 📏 Circularity filter  
   - ...  

Ideal for datasets with structured or industrial shapes.  
You can then transfer the detected shape or BBox into the **extra shapes** section.

---

### 🌱 Available predefined index & Vegetation Indices  
These indices are computed from the RGB channels:

#### **Base Excess / Difference Indices**
- `ExG = 2G - R - B`  
- `ExR = 1.4R - G`  
- `ExB = 1.4B - G`  
- `ExGR = G - R`  
- `ExRB = R - B`  
- `ExGB = G - B`  

#### **Normalized Difference Indices**
- `NDI_GR = (G - R) / (G + R)`  
- `NDI_BR = (B - R) / (G + R)`  
- `NDI_GB = (G - B) / (G + B)`  
- `NRB = (R - B) / (R + B)`  
- `NRGDI = (R - G) / (R + G)`  
- `NRBDI = (R - B) / (R + G + B)`  
- `NGRDI_mod = (G - R) / (G + R + B)`  
- `NGBDI_mod = (G - B) / (G + B + R)`  

#### **Channel Ratios**
- `GR_Ratio = G / R`  
- `GB_Ratio = G / B`  
- `RG_Ratio = R / G`  
- `RB_Ratio = R / B`  
- `BG_Ratio = B / G`  
- `BR_Ratio = B / R`  

#### **Channel Fractions**
- `G_Fraction = G / (R + G + B)`  
- `R_Fraction = R / (R + G + B)`  
- `B_Fraction = B / (R + G + B)`  

#### **Additional Normalized Indices**
- `NGI = G / (R + G + B)`  
- `NRGI = (R + G) / (R + G + B)`  
- `NRBI = (R + B) / (R + G + B)`  
- `NEGI = (G - R - B) / (R + G + B)`  
- `NERI = (R - G - B) / (R + G + B)`  
- `NEBI = (B - G - R) / (R + G + B)`  

#### **Vegetation-Focused Modified Indices**
- `MGRVI = (G - R) / (G + R + 2B)`  
- `MGBVI = (G - B) / (G + B + 2R)`  
- `MRGVI = (R - G) / (R + G + 2B)`  

---

# 4️⃣ Saving & Exporting 💾  
- All annotations are saved in **YOLO format**  
- Output `.txt` files are written directly in the `labels/` folder  
- Compatible with YOLOv5–YOLOv9, RT-DETR, SAM pipelines, and others

---

# 5️⃣ Feature Summary ✨

### 5.1 Manual Annotation Tools  
- Create and edit BBoxes  
- Create and edit segmentation masks  
- Class assignment  
- Zoom and pan  
- Multi-mode precision editing  

### 5.2 Assisted Tools  
- YOLO predictions  
- Machine Vision shape detection  
- IOU-based “Extra” matching for missing objects  

### 5.3 Visualization Tools  
- Layer toggles (GT / Pred / Extra)  
- Transparency sliders  
- Clean two-panel layout  

---

# 6️⃣ Roadmap ⏳  
- YOLO-based **mask prediction**  
- Advanced segmentation editing utilities  
- Upgrade dataset description panels

---

# 7️⃣ Contributing 🤝  
Issues, suggestions, and pull requests are welcome.  
Feel free to reach out for improvements or new features.

---

# 8️⃣ License 📄  
Use it, upgrade it, make it yours.
