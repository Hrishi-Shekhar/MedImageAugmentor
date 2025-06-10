Image Augmentation Pipeline

A complete data augmentation pipeline for skin cancer images, including:

1. Cropping images using YOLO annotations or superpixel masks

2. Background removal from cropped images

3. Web scraping of medical skin backgrounds

4. Overlaying cropped foregrounds onto backgrounds

5. Organized file structure for scalability and modular use

Folder structure-

synthetic-image-augmentation/
├── data/
│   └── dataset-1_augmentation/
│       ├── input/
│       │   ├── images/                     # Original input images
|       |   ├── json_input/                 # Json annotations
│       │   └── labels/                     # YOLO annotations (txt files)
│       │
│       ├── backgrounds/
│       │   ├── user_generated/             # User provided backgrounds
│       │   └── web_scraping/               # Web-scraped backgrounds
│       │
│       ├── intermediate/
│       │   ├── cropped/                    # Cropped object images (from YOLO/masks)
│       │   └── cropped_nobg/               # Cropped images after background removal
│       │
│       └── output/
│           ├── composites/                 # Final composite images
│           ├── annotations/                # New annotations 
|           └── coco_annotations.json       # Annotations in json format
│                         
│
├── scripts/
│   ├── cropping_imgs.py                    # Cropping logic (YOLO/mask)
│   ├── bg_removal.py                       # Background removal
|   ├── bg_extraction_web_scraping.py       # Web Scraping
│   ├── overlay.py                          # Overlay cropped objects on backgrounds
|   ├── label_conversion.py                 # Converting json/xml input to yolo format
|   ├── yolo_to_json.py                     # Converting yolo output to json format
|   └── yolo_to_mask.py                     # Converting yolo output to masks
│
├── main.py                                 # Main orchestrator script  
│
├── README.md                               # Project documentation
├── pipeline.log                            # Logs of the entire pipeline
├── config.yaml                             # Config file for central control
├── requirements.txt                        # List of dependencies
└── .gitignore                              # Files/folders to ignore in Git

Getting Started-

1. Clone the repository-

    git clone https://github.com/Hrishi-Shekhar/Augmentation.git
    cd Augmentation

2. Install Dependencies-

    pip install -r requirements.txt

3. Run the pipeline-

    python main.py



