.npz is a numpy file type for storing multiple numpy arrays in a single file
Steps taken:
1. install a good dataset
2. convert the dataset into 

# Problem: I tried to run python3 scripts/to_npz.py
What I learned:
- When you run python scripts/to_npz.py, Python assumes your “current package” is scripts/ and looks for imports starting from there.
- It does not automatically treat the parent folder (emotion-cnn/) as part of the search path unless you run the script from the project root.
- sys.path[0] = the directory containing the script you run directly.
- That means sibling files (like dataset.py) aren’t automatically importable unless they’re in the same directory or on sys.path.

How it was fixed:
- To run from the project root and use -m to run modules: python3 -m scripts.to_npz
Alternate Solutions:
- Or, manually add the project root to sys.path in the script (quick fix, but less clean).
- Or, make your code a package with __init__.py files and always import using the package name.

# Question: Should I implement augmentation into the dataset?
Short answer: yes—but put augmentation in the loader/training pipeline, not in the NPZ preconvert. Keep val/test strictly unaugmented. On tiny 48×48 faces, use light, label‑preserving aug; heavy warps often hurt.

