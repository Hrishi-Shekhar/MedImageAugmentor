from bing_image_downloader import downloader

def download_backgrounds(keyword, limit, output_dir):
    downloader.download(
        keyword,
        limit=limit,
        output_dir=output_dir,
        adult_filter_off=True,
        force_replace=False,
        timeout=60
    )
    print(f"Downloaded {limit} backgrounds with keyword: {keyword}")


