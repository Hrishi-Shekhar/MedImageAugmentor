# 🧠 Synthetic Image Augmentation Pipeline

A complete, modular, and production-ready pipeline for **medical image augmentation**, designed to automate and scale the creation of synthetic training data using object cropping, background removal, web-scraped or generated backgrounds, and annotation conversion.

This pipeline is highly configurable via `config.yaml` and supports YOLO and COCO formats. It is well-suited for use in computer vision tasks such as **segmentation, classification, and detection**, especially in medical imaging domains.

---

##  Features

-  Crop objects from images using YOLO or JSON/XML annotations
-  Remove background from cropped regions using pretrained models
-  Automatically download relevant backgrounds via web scraping
-  Overlay cropped objects onto user-defined or scraped backgrounds
-  Organize outputs for easy dataset training and extension
-  Convert annotations to **YOLO** and **COCO** formats
-  Generate binary segmentation masks from YOLO annotations
-  Modular, scalable file structure and centralized config

---

## 📂 Required Input Dataset Structure

Before running the pipeline, you must provide the following minimum structure inside your dataset folder (as defined in `config.yaml > data_root`):

```text
data/
└── your_dataset_name/                    # <- Defined in config.yaml (data_root)
    ├── images/                           #  Required: Input images (.jpg / .png)
    ├── class_names.txt                   #  Required: One class per line (must match annotations)
    ├── labels/                           #  Optional: YOLO-format .txt files 
    ├── json/                             #  Optional: JSONL annotations (e.g., CVAT format)
    ├── xml/                              #  Optional: Pascal VOC .xml annotations
    └── test/                             #  Optional: Extra test set (copied to input/test)
```

Important:

You must provide at least one of the following annotation formats:

1. labels/ (YOLO .txt files)

2. json/ (JSONL annotations)

3. xml/ (Pascal VOC XML)

The pipeline will:

Use YOLO annotations directly if labels/ is present.

Otherwise, it will automatically convert from json/ or xml/ into YOLO format.

On the first run, the pipeline will automatically organize this data into the following structure under `input/`, `intermediate/`, and `output/` folders.

---

## 📁 Project Folder Structure (Post Run)

```text
synthetic-image-augmentation/
├── data/
│ └── your_dataset_name/
│ ├── input/
│ │ ├── images/ # Organized input images
│ │ ├── labels/ # YOLO-format annotations
│ │ └── test/ # Test data (optional)
│
│ ├── backgrounds/
│ │ ├── user/ # User-provided backgrounds
│ │ └── web/ # Web-scraped backgrounds
│
│ ├── intermediate/
│ │ ├── cropped/ # Cropped objects from original images
│ │ └── cropped_nobg/ # Foregrounds after background removal
│
│ └── output/
│ ├── composites/ # Final composite images
│ ├── annotations/ # Updated YOLO annotations
│ ├── masks/ # Binary masks
│ └── coco_annotations.json # Converted COCO annotations

```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Hrishi-Shekhar/synthetic-image-augmentation.git
cd synthetic-image-augmentation
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Your Data
Place your input data as shown in the "Required Dataset Structure" above.

### 4. Run the Pipeline

```bash
python main.py
```

## Outputs
After running, your output/ directory will contain:

Composite images with generated backgrounds

Updated YOLO annotations

COCO-style annotations

Binary segmentation masks

Copied originals for reference

## Use Cases
Medical image classification and segmentation

Data augmentation for small or imbalanced datasets

Synthetic data generation for robust model training

Domain-specific background enhancement


