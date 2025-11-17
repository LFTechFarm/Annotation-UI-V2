# 📘 Annotation-UI-V2  
### A Lightweight, Powerful Annotation Tool for Bounding Boxes & Masks

**Annotation-UI-V2** is a Python/Tkinter-based graphical interface designed to **speed up dataset annotation** for computer vision tasks.  
It supports **bounding boxes**, **segmentation masks**, **machine-vision helpers**, and **YOLO-based predictions**, making it ideal for building high-quality training datasets.

Feel free to use, modify, or contact me for improvements!

---

# 1️⃣ Usage — Step-by-Step Tutorial 🖥

## 1.1 Launching the Application  
You can start Annotation-UI-V2 in two ways:

- Double-click the executable  
**or**  
- Run the Python script directly  

The interface should open like this:  
*(insert screenshot here)*

---

## 1.2 Selecting a Dataset 📂

Click **“Select Parent Folder”** (top-left) and choose your dataset folder.

Your dataset must follow the YOLO directory structure:


If you only have images, create an empty `labels/` folder.

---

## 1.3 Navigating Images 🔍

- Use the **slider** or **arrow buttons** to browse images  
- Toggle visibility of:
  - Ground Truth (GT)
  - Predictions
  - Extra boxes  
- Adjust **transparency** with sliders  
- Switch between **Bounding Box** and **Mask** tasks using the tabs  

---

## 1.4 Annotation Modes 🛠

Each mode activates a different edit action:

| Mode | Description |
|------|-------------|
| ✏️ **Draw** | Create a new bounding box |
| ✂️ **Move / Resize** | Modify the current BBox |
| 🗑️ **Delete** | Remove a BBox permanently |
| 🔀 **Change Class** | Change the class of the selected object |
| ✨ **Validate** | Transfer predicted boxes to GT |
| ⚒️ **Mask Edit** | Click mask points to adjust or reshape |

---

## 1.5 AI Predictions with YOLO 🤖

If you have a trained YOLO model:

1. Click **Import Weights** and load `best.pt`  
2. Click **Predict Image**  
3. Show predictions by enabling checkboxes  
4. Validate predictions into GT:
   - Individually  
   - Or all at once  
5. Use **Extra** to detect missing boxes via IOU  

This greatly accelerates dataset labeling.

---

## 1.6 Machine Vision Tools 📷

These tools help automate annotation using classical CV algorithms.

Workflow:

1. Select the image index  
2. Generate a mask by selecting a threshold  
3. Detect shapes:  
   - ⚪ Circle  
   - 🔺 Triangle  
   - ⬛ Rectangle  
   - 🔷 Polygon  
   - 📏 Circularity filtering  

Useful for datasets with consistent geometric shapes.

---

## 1.7 Saving & Exporting 💾

- Save your edited annotations anytime  
- Outputs are stored in **YOLO-compatible `.txt` files**  
- Mask edits are exported according to the selected format  

---

# 2️⃣ Features Overview ✨

### 2.1 YOLO Dataset Navigation  
- Load full datasets instantly  
- Displays image ↔ label pairing automatically  

### 2.2 Bounding Box & Mask Editing  
- Intuitive drawing tools  
- Point-based mask refinement  
- Class management & reassignment  

### 2.3 Transparency Controls  
- Adjust opacity of:
  - Ground Truth  
  - Model Predictions  
  - Extra/secondary boxes  

### 2.4 Assisted Annotation  
- YOLO model integration  
- Machine vision shape detection  
- IOU-based missing-object discovery  

### 2.5 Pan & Zoom  
- Large image support  
- Arrow-key panning  
- Smooth zooming  

### 2.6 Export  
- Saves in **standard training formats**  
- Compatible with YOLOv5/6/7/8/9, RT-DETR, SAM-based pipelines, etc.

---

# 3️⃣ Roadmap ⏳

- 🔜 YOLO-based **mask prediction**  
- 🔜 Advanced segmentation editing tools  
- 🔜 Plugin architecture for user-defined CV pipelines  

---

# 4️⃣ Contributing 🤝

Issues, suggestions, and pull requests are welcome!  
Feel free to reach out for collaborations or feature ideas.

---

# 5️⃣ License 📄

Specify your license here (MIT, Apache 2.0, GPL, etc.)

