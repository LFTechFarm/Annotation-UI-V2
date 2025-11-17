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
*(insert screenshot)*

---

## 1.2 Load Your Dataset 📂  
Click **“Select Parent Folder”** (top-left).  
Your dataset must follow YOLO format:

dataset/
images/
labels/



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
Used for all manual annotation work.

### 🟣 Right Panel = **Helpers (Dataset Info, YOLO, Machine Vision)**  
Used for automation, predictions, and shape detection.

This separation keeps annotation clean and efficient.

---

# 2️⃣ Left Panel — Task Tools

## 2.1 Image Navigation & Display  
- Move between images with the **slider** or **arrow buttons**  
- Zoom and pan to inspect details  
- Toggle visibility of:
  - Ground Truth (GT)
  - Predictions
  - Extra boxes  
- Adjust transparency with sliders  

---

## 2.2 Annotation Modes — Bounding Boxes  
Active when **BBox** task is selected:

| Mode | Description |
|------|-------------|
| ✏️ **Draw** | Create a new bounding box |
| ✂️ **Move / Resize** | Adjust the selected box |
| 🗑️ **Delete** | Remove a box |
| 🔀 **Change Class** | Modify the object class |
| ✨ **Validate** | Move predicted boxes to GT |

---

## 2.3 Annotation Modes — Masks  
Active when **Mask** task is selected:

| Mode | Description |
|------|-------------|
| ✏️ **Draw Mask** | Create a new segmentation mask |
| ⚒️ **Edit Mask** | Move contour points to refine shape |
| 🗑️ **Delete Mask** | Remove the mask |
| 🔀 **Change Class** | Modify the object class |
| ✨ **Validate** | Transfer predicted mask to GT |

---

# 3️⃣ Right Panel — Helper Sections

## 3.1 Dataset Description  
Provides an overview of:
- Current image index  
- Total images  
- Loaded classes  
- Available label files  

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
Classical CV algorithms to automatically detect geometric shapes.

### Workflow:
1. Select image index  
2. Generate a **binary mask** using thresholding  
3. Apply shape detection:
   - ⚪ Circle  
   - 🔺 Triangle  
   - ⬛ Rectangle  
   - 🔷 Polygon  
   - 📏 Circularity filter  

Ideal for datasets with structured or industrial shapes.

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
- Plugin system for custom CV scripts  

---

# 7️⃣ Contributing 🤝  
Issues, suggestions, and pull requests are welcome.  
Feel free to reach out for improvements or new features.

---

# 8️⃣ License 📄  
Specify your license here (MIT, Apache, GPL, etc.)

