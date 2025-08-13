import os, numpy as np
from PIL import Image

from dataset import EMOTIONS

def read_image_as_tensor(path, size = (48,48)):
    img = Image.open(path).convert("L") #"L" only stores illuminescence, "P" mode means it is palletised.
    if img.size != size:
        img = img.resize(size, Image.Resampling.BILINEAR)
        x = np.asarray(img, dtype = np.float32) / 255.0 #divide to normalize
        return x[None, ...] # from (48, 48) to (1,48,48), this is because CNN will later expect (N - batch size, C - channels (1 for grayscale 3 for rgb), H height, W -width), later we will stack many (1, 48,48 ) to get the desired form.



def pack_split(in_root, out_path):
    X, y = [], []

    for i, emotion in enumerate(EMOTIONS):
        emotion_dir = os.path.join(in_root, emotion)
        if not os.path.isdir(emotion_dir):
        # Skip missing classes (or raise if you want strictness)
            continue    
        for filename in os.listdir(emotion_dir):
            if filename.lower().endswith((".jpg", "jpeg", "png")):
                path = os.path.join(emotion_dir, filename)
                try:
                    X.append(read_image_as_tensor(path))
                    y.append(i)
                except Exception as e:
                    # Skip corrupt/unreadable files, but note them
                    print(f"Warning: failed to load {path}: {e}")
    X = np.stack(X, axis = 0).astype(np.float32)    # np.stack with argument axis is more deterministic even though in this example np.array wouldve performed the same. 
                                                    # 32 bit float is the standard
    y = np.array(y, dtype=np.int64)                 #64 bit int is standard
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True) # makes a dir if it doesnt exist, for safety purposes
    np.savez_compressed(out_path, X = X, y= y)

pack_split('data/test', 'data/test48.npz')
pack_split('data/train', 'data/train48.npz')