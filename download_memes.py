"""Download placeholder reaction images for PhoneWatch meme alerts."""

from __future__ import annotations

import urllib.error
import urllib.request
from pathlib import Path


MEME_DIR = Path(__file__).resolve().parent / "memes"
REACTION_IMAGES = [
    ("shocked_reaction", "https://placehold.co/640x480/png?text=Shocked"),
    ("side_eye_reaction", "https://placehold.co/640x480/png?text=Side+Eye"),
    ("facepalm_reaction", "https://placehold.co/640x480/png?text=Facepalm"),
    ("not_again_reaction", "https://placehold.co/640x480/png?text=Not+Again"),
    ("seriously_reaction", "https://placehold.co/640x480/png?text=Seriously%3F"),
    ("caught_reaction", "https://placehold.co/640x480/png?text=Caught+In+4K"),
    ("dramatic_reaction", "https://placehold.co/640x480/png?text=Dramatic+Pause"),
    ("put_it_down_reaction", "https://placehold.co/640x480/png?text=Put+It+Down"),
]


def download_memes() -> list[Path]:
    MEME_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for name, url in REACTION_IMAGES:
        target = MEME_DIR / f"{name}.png"
        try:
            with urllib.request.urlopen(url, timeout=12) as response:
                target.write_bytes(response.read())
            downloaded.append(target)
            print(f"Downloaded {name}: {url} -> {target}")
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            print(f"Failed to download {name} from {url}: {exc}")
    return downloaded


if __name__ == "__main__":
    files = download_memes()
    print(f"Downloaded {len(files)} meme images into {MEME_DIR}")
