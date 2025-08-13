# augment.py
#copy-pasted from chatgpt, later will be read through.
import numpy as np

def flip_lr(x):
    # x: (C,H,W)
    if np.random.rand() < 0.5:
        x = x[..., ::-1]
    return x

def translate(x, max_shift=3):
    C,H,W = x.shape
    pad = max_shift
    canvas = np.zeros((C, H+2*pad, W+2*pad), dtype=x.dtype)
    canvas[:, pad:pad+H, pad:pad+W] = x
    dy = np.random.randint(0, 2*pad+1)
    dx = np.random.randint(0, 2*pad+1)
    return canvas[:, dy:dy+H, dx:dx+W]

def brightness_contrast(x, b_range=0.10, c_range=0.10):
    b = (np.random.rand()*2-1) * b_range       # [-b_range, b_range]
    c = 1.0 + ((np.random.rand()*2-1) * c_range)
    y = x * c + b
    return np.clip(y, 0.0, 1.0)

def gaussian_noise(x, sigma=0.03, p=0.4):
    if np.random.rand() < p:
        x = x + np.random.randn(*x.shape).astype(x.dtype)*sigma
        x = np.clip(x, 0.0, 1.0)
    return x

def cutout(x, max_frac=0.25, p=0.4):
    if np.random.rand() >= p: 
        return x
    C,H,W = x.shape
    fh, fw = int(H*max_frac*np.random.rand()), int(W*max_frac*np.random.rand())
    cy, cx = np.random.randint(0, H), np.random.randint(0, W)
    y0, y1 = max(0, cy-fh//2), min(H, cy+fh//2)
    x0, x1 = max(0, cx-fw//2), min(W, cx+fw//2)
    x[:, y0:y1, x0:x1] = x.mean()  # or 0.0
    return x

def augment(x):
    # order matters a bit; keep it gentle for 48×48
    x = flip_lr(x)
    x = translate(x, max_shift=3)
    x = brightness_contrast(x, b_range=0.08, c_range=0.08)
    x = gaussian_noise(x, sigma=0.02, p=0.3)
    x = cutout(x, max_frac=0.20, p=0.3)
    return x
