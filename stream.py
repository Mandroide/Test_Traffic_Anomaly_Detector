import time


def download_video(url: str) -> None:
    """Download a video."""
    start_time = time.time()

    end_time = time.time()
    total_time = time.gmtime(end_time - start_time)
    # Total runtime taken time.strftime("%H:%M:%S", )
    print(f"Runtime of the program is {time.strftime('%H:%M:%S', total_time)}")
