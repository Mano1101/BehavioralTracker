"""Installer artwork from resources/icon.png: app.ico, welcome.bmp (164x314), header.bmp (150x57)."""
import os
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(__file__)
VERSION = "1.0.0"
icon = Image.open(os.path.join(ROOT, "resources", "icon.png")).convert("RGBA")
icon.save(os.path.join(OUT, "app.ico"), sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])


def font(size, bold=True):
    for p in ("/usr/share/fonts/truetype/google-fonts/Poppins-Bold.ttf" if bold else
              "/usr/share/fonts/truetype/google-fonts/Poppins-Regular.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "C:/Windows/Fonts/segoeuib.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def gradient(w, h, top=(22, 52, 140), bottom=(18, 170, 160)):
    img = Image.new("RGB", (w, h))
    px = img.load()
    for y in range(h):
        t = y / max(1, h - 1)
        c = tuple(int(round(top[i] + (bottom[i] - top[i]) * t)) for i in range(3))
        for x in range(w):
            px[x, y] = c
    return img


# welcome / finish page side panel
w = gradient(164, 314)
ic = icon.resize((124, 124), Image.LANCZOS)
w.paste(ic, (20, 40), ic)
d = ImageDraw.Draw(w)
d.text((82, 190), "Behavioral", fill="white", font=font(19), anchor="mm")
d.text((82, 214), "Tracker", fill=(160, 255, 235), font=font(19), anchor="mm")
d.text((82, 250), f"Version {VERSION}", fill="white", font=font(12, bold=False), anchor="mm")
w.save(os.path.join(OUT, "welcome.bmp"))
# header (top-right of the inner pages)
hd = Image.new("RGB", (150, 57), "white")
ic = icon.resize((50, 50), Image.LANCZOS)
hd.paste(ic, (96, 3), ic)
hd.save(os.path.join(OUT, "header.bmp"))
print("assets written to", OUT)
