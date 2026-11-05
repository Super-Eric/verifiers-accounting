"""Record results/arch-kernels-v1/film/index.html to an MP4 (1080x1350, 30 fps) for X.

Every frame is rendered exactly with the page's window.__frame(t) hook, so the video does not depend on
playback speed. Needs Playwright (uses the installed Chrome) and ffmpeg.

Usage: uvx --with playwright python scripts/record_film.py <frames_dir> <out.mp4> [t1 t2 ...]   (times = stills only)
"""

import subprocess
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

PAGE = Path(__file__).resolve().parent.parent / "results" / "arch-kernels-v1" / "film" / "index.html"
FPS, DUR = 30, 46.0


def main(frames: Path, out: Path, stills: list[float]) -> None:
    frames.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        page = browser.new_page(viewport={"width": 1200, "height": 1500})
        page.goto(PAGE.as_uri())
        page.evaluate("document.fonts.ready")
        page.wait_for_timeout(1500)
        times = stills or [i / FPS for i in range(int(DUR * FPS))]
        for i, t in enumerate(times):
            page.evaluate(f"window.__frame({t})")
            page.locator("#frame").screenshot(path=str(frames / f"f{i:05d}.png"))
        browser.close()
    if not stills:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", str(frames / "f%05d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", str(out)],
                       check=True)
        print("wrote", out)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), [float(x) for x in sys.argv[3:]])
