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



Convolutional Layer 
- Puropse: Recognizes spatial features such as edges, textures or facial patterns 
- Input:
    - x (N, C, H, W) training/testing data
    - w (F, C, kH, kW) filters
    - b (F, ) biases
- How it works:
    -  Forward: 
        1. use im2col to turn each image into the collection of flattened patches
        2. multiply each patch with weights
        3. Reshape into (F, C, kH, kW)
    - Backward:
    get dout (N, F, H_out, W_out) from previous layer, this is δL/δout
    remember out = X_cols @ W_col.T 
    δL/δx = δL/δout * δout/δx = dout * W_col.T
    same done for δL/δw and δb
         1. Compute gradients wrt:
            * dx (input) 
            * dw (weights)
            * db (biases)
        2. use col2im to turn it back into a original shape

ReLU Layer
- Purpose: Adds non-linearity so that the network can learn complex functions
* Forward: out = max(0, x) element vise
* Backward: Pass gradient only when x>0 


Pooling Layer
Purpose: Reduce spatial size, add translation invariance 
* Forward: For each region, take max value (or average for avg-pooling)
* Backward: Pass gradient only to the position that was max in the forward pass

Fully Connected Layer
Purpose: Map extracted features to the final predictions
* Forward:
    - Flatten input to (N,D)
    - Compute out = x @ W + b
*Backward: dx, DW, db with standard matrix calculus

Soft Max Layer
- Purpose: standardize the probabilities s.t. Σp =1 while keeping the gradient non-vanishing. 
* Forward: u alr know the formula, compute the loss with respect to the labels
* Backward: dscores = p - y_one_hot

CNN we are building:
Input (48x48 grayscale img) → 
Conv → ReLU → Pool →
Conv → ReLU → Pool →
FC → ReLU →
FC → SoftMax → output

Why this order works well
It’s a progressive abstraction pipeline:
Local to global: Start with fine pixel-level features, end with whole-image semantics.
Non-linear expansion: ReLU after each conv lets the network build more complex features from simple ones.
Dimensionality control: Pooling keeps computation in check and avoids overfitting by discarding minor details.
Decision making: FC layers combine all learned features to make the final classification.


Numpy Advanced Indexing:

In Numpy we can pass arrays during slices. Through this, easily mutate original multidimensional tensors, and objects represented through it.

Ex 1 from code:
