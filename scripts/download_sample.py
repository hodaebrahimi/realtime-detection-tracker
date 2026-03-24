"""Download a short demo video into data/sample.mp4."""

from __future__ import annotations

import argparse
import pathlib
import urllib.request

DEFAULT_URL = "https://videos.pexels.com/video-files/855564/855564-hd_1280_720_30fps.mp4"


def download_sample(url: str = DEFAULT_URL, output_path: str = "data/sample.mp4") -> pathlib.Path:
    """Download a sample video file from a public URL."""
    destination = pathlib.Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)
    return destination


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download sample video for demo.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Public video URL.")
    parser.add_argument("--output", default="data/sample.mp4", help="Output path.")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for sample downloader."""
    args = parse_args()
    output = download_sample(url=args.url, output_path=args.output)
    print(f"Sample video downloaded to: {output}")


if __name__ == "__main__":
    main()
