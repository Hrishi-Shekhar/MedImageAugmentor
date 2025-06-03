from bing_image_downloader import downloader
import logging
from PIL import Image
from pathlib import Path

log = logging.getLogger(__name__)

def download_backgrounds(keyword, limit, output_dir,width=None,height=None):
    try:
        downloader.download(
            keyword,
            limit=limit,
            output_dir=str(output_dir),
            adult_filter_off=True,
            force_replace=False,
            timeout=60
        )
        log.info(f"Downloaded {limit} backgrounds with keyword: '{keyword}'")

        if width and height:
            downloaded_path = Path(output_dir) / keyword
            for img_file in downloaded_path.glob("*.[jp][pn]g"):
                try:
                    img = Image.open(img_file).convert("RGB")
                    img = img.resize((width, height), Image.ANTIALIAS)
                    img.save(img_file)
                except Exception as e:
                    log.warning(f"Failed to resize {img_file}: {e}")
            log.info(f"Resized all images in '{downloaded_path}' to {width}x{height}")

    except Exception as e:
        log.error(f"Failed to download images for keyword '{keyword}': {e}")

