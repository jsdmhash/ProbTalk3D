"""Draws the ProbTalk3D Studio app icon (waveform + smile) with Pillow."""
import math, sys
import numpy as np
from PIL import Image, ImageDraw
S = 4 * 512                                    # draw large, shrink for smooth edges
yy, xx = np.mgrid[0:S, 0:S] / S
t = (xx + yy) / 2
bg = (np.array([26, 20, 64]) * (1 - t[..., None]) + np.array([7, 10, 19]) * t[..., None])
glow = np.exp(-(((xx - .5) ** 2 + (yy - .45) ** 2) / (2 * .22 ** 2)))[..., None]
bg = bg * (1 - .45 * glow) + np.array([124, 108, 255]) * .45 * glow
img = Image.fromarray(bg.clip(0, 255).astype("uint8"))
stops = [(0, (124, 108, 255)), (.55, (228, 92, 240)), (1, (54, 214, 195))]
def grad(v):
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        if v <= b:
            f = (v - a) / (b - a)
            return np.array(ca) * (1 - f) + np.array(cb) * f
    return np.array(stops[-1][1])
g = np.array([[grad(v) for v in row] for row in ((xx[::16, ::16] + yy[::16, ::16]) / 2)])
gimg = Image.fromarray(g.astype("uint8")).resize((S, S), Image.BILINEAR)
mask = Image.new("L", (S, S), 0); m = ImageDraw.Draw(mask)
k = S / 512
for i, x in enumerate(range(96, 432, 24)):
    tt = (x - 256) / 160
    h = 40 + 150 * math.exp(-6 * tt * tt) * (0.6 + 0.4 * abs(math.sin(i * 1.7)))
    m.rounded_rectangle([(x - 8) * k, (230 - h / 2) * k, (x + 8) * k, (230 + h / 2) * k], radius=8 * k, fill=255)
img.paste(gimg, (0, 0), mask)
d = ImageDraw.Draw(img)
pts = [((1 - u) ** 2 * 150 + 2 * (1 - u) * u * 256 + u * u * 362,
        (1 - u) ** 2 * 360 + 2 * (1 - u) * u * 430 + u * u * 360) for u in np.linspace(0, 1, 400)]
r = 9 * k
for x, y in pts:
    d.ellipse([x * k - r, y * k - r, x * k + r, y * k + r], fill=(54, 214, 195))
out = sys.argv[1] if len(sys.argv) > 1 else "icon.png"
img.resize((512, 512), Image.LANCZOS).save(out)
print("icon saved:", out)
