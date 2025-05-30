from bing_image_downloader import downloader
import logging

log = logging.getLogger(__name__)

def download_backgrounds(keyword, limit, output_dir):
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
    except Exception as e:
        log.error(f"Failed to download images for keyword '{keyword}': {e}")

