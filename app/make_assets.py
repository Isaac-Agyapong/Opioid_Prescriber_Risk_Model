"""Generate the app's hero banner and brand logo (run once; outputs are committed in app/assets/)."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path(__file__).resolve().parent / "assets"
OUT.mkdir(exist_ok=True)


def hero(w=1800, h=300, s=2):
    W, H = w * s, h * s
    img = Image.new("RGB", (W, H))
    px = img.load()
    left, right = (10, 28, 48), (11, 72, 88)
    for x in range(W):
        t = (x / W) ** 1.6
        c = tuple(round(a + (b - a) * t) for a, b in zip(left, right))
        for y in range(H):
            px[x, y] = c
    img = img.convert("RGBA")
    over = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    # architectural light lines on the right half
    for i in range(26):
        x0 = int(W * 0.55) + i * 38 * s
        d.line([(x0, H), (x0 + 260 * s, 0)], fill=(94, 234, 212, 26 if i % 3 else 60), width=s)
    for i in range(12):
        x0 = int(W * 0.72) + i * 60 * s
        d.line([(x0, 0), (x0, H)], fill=(125, 211, 252, 22), width=s)
    # translucent glass panels with bright edges
    for x0, w0, top, skew, alpha in [(0.74, 90, 0.10, 60, 34), (0.80, 130, -0.05, 80, 44), (0.89, 110, 0.22, 70, 38)]:
        x, ww, t, k = int(W * x0), w0 * s, int(H * top), skew * s
        panel = [(x, H), (x + k, t), (x + k + ww, t), (x + ww, H)]
        d.polygon(panel, fill=(94, 234, 212, alpha))
        d.line([panel[0], panel[1]], fill=(153, 246, 228, 150), width=2 * s)
        d.line([panel[1], panel[2]], fill=(153, 246, 228, 110), width=2 * s)
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse([int(W * 0.7), -H, int(W * 1.1), H], fill=(20, 184, 166, 70))
    over = Image.alpha_composite(over, glow.filter(ImageFilter.GaussianBlur(90 * s)))
    img = Image.alpha_composite(img, over.filter(ImageFilter.GaussianBlur(0.6 * s)))
    img.convert("RGB").resize((w, h), Image.LANCZOS).save(OUT / "hero.png", optimize=True)


def logo(s=4):
    W, H = 360 * s, 80 * s
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # two overlapping diamonds (teal + cyan)
    cx, cy, r = 34 * s, 40 * s, 22 * s
    d.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)], fill=(20, 184, 166, 255))
    cx2 = cx + 14 * s
    d.polygon([(cx2, cy - r + 8 * s), (cx2 + r - 8 * s, cy), (cx2, cy + r - 8 * s), (cx2 - r + 8 * s, cy)],
              fill=(186, 230, 253, 235))
    try:
        font = ImageFont.truetype("segoeuib.ttf", 26 * s)
    except OSError:
        font = ImageFont.load_default()
    d.text((78 * s, 10 * s), "Prescribing", font=font, fill=(255, 255, 255, 255))
    d.text((78 * s, 40 * s), "Insights", font=font, fill=(255, 255, 255, 255))
    img.resize((360, 80), Image.LANCZOS).save(OUT / "logo.png")


if __name__ == "__main__":
    hero()
    logo()
    print("assets written")
