Image Augmentation Pipeline

A complete data augmentation pipeline for skin cancer images, including:

1. Cropping images using YOLO annotations or superpixel masks

2. Background removal from cropped images

3. Web scraping of medical skin backgrounds

4. Overlaying cropped foregrounds onto backgrounds

5. Organized file structure for scalability and modular use

Folder structure-

Skin-Cancer-Data-Augmentation/
├── data/
│   └── Skin_cancer_augmentation/
│       ├── input/
│       │   ├── images/                     # Original input images
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
│           └── annotations/                # New annotations 
│                         
│
├── scripts/
│   ├── cropping_imgs.py                    # Cropping logic (YOLO/mask)
│   ├── bg_removal.py                       # Background removal
|   ├── bg_extraction_web_scraping.py       # Web Scraping
│   └── overlay.py                          # Overlay cropped objects on backgrounds
│
├── main.py                                 # Main orchestrator script  
│
├── README.md                               # Project documentation
├── requirements.txt                        # List of dependencies
└── .gitignore                              # Files/folders to ignore in Git

Getting Started-

1. Clone the repository-

    git clone https://github.com/Hrishi-Shekhar/Augmentation.git
    cd Augmentation

2. Install Dependencies-

    pip install -r requirements.txt



