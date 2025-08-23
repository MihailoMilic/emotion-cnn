
# Diary 1
.npz is a numpy file type for storing multiple numpy arrays in a single file
Steps taken:
1. install a good dataset
2. convert the dataset into 

## Problem: I tried to run python3 scripts/to_npz.py
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


# Diary 2
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

# Diary 3
Numpy Advanced Indexing:

In Numpy we can pass arrays during slices. Through this, easily mutate original multidimensional tensors, and objects represented through it.


# Diary 4
Finished coding the model.

I am having dificulties with training. the loss is consistantly jumping around 2 across all batches and the precision and accuracy is 0.

I've found several logical bugs in my code which are now fixed.

The loss isn't budging still. 

I constructed a gradient checker for the params (if exist) and for the inputs of layers.
Results:
[IN ] Layer 00 Conv2D                ok=True  max_err=1.642e-03
[PAR] Layer 00 Conv2D                param[0]   ok=True  max_err=3.458e-06
[PAR] Layer 00 Conv2D                param[1]   ok=True  max_err=7.629e-06
[IN ] Layer 01 ReLU                  ok=True  max_err=2.855e-11
[IN ] Layer 02 MaxPool2D             ok=False  max_err=1.341e+00 <----------\
[IN ] Layer 03 Conv2D                ok=True  max_err=2.015e-07              \
[PAR] Layer 03 Conv2D                param[0]   ok=True  max_err=9.963e-07      The problem is the MaxPool2D layer
[PAR] Layer 03 Conv2D                param[1]   ok=True  max_err=1.907e-06    /
[IN ] Layer 04 ReLU                  ok=True  max_err=1.876e-11              /
[IN ] Layer 05 MaxPool2D             ok=False  max_err=1.157e+00 <----------/
[IN ] Layer 06 Flatten               ok=True  max_err=1.023e-11
[IN ] Layer 07 Linear                ok=True  max_err=2.690e-08
[PAR] Layer 07 Linear                param[0]   ok=True  max_err=2.790e-08
[PAR] Layer 07 Linear                param[1]   ok=True  max_err=2.384e-07
[IN ] Layer 08 ReLU                  ok=True  max_err=2.148e-12
[IN ] Layer 09 Linear                ok=True  max_err=5.545e-08
[PAR] Layer 09 Linear                param[0]   ok=True  max_err=2.908e-08
[PAR] Layer 09 Linear                param[1]   ok=True  max_err=1.192e-07

MaxPool2D is causing problems. I assume i made a logical error in coding the backward function.  
MaxPool2D passed the standalone exam. (True, 0.0029653310775756836)
Tried syntetic input, both feed forward and backward work fine. 

The key problem was in my convolutional layer. im2col helper function was returning a column with incorrect shape. 


# Timeline of fixes:
1. Loss hovering around 2, model is randomly guessing
    - Next Step: Tiny subset overfit
2. Fix gradient check
    - Was calling backward() instead of forward. Correctly defined the Loss function as L = sum(out *dout)
    - Allowing the gradient check to be impactful
3. The checker was ran for each layer with the raw input, even though in our model only the first layer gets the raw input.
    - Fixed with an array of activations for each layer.
4. No parameter gradient check
    - So far only dx was checked and not dW or db for linear and Convolutional.
    - Since in later phases all param checks based it allowed us to isolate the issue.
5. MaxPool2D bugs
    - used w + stride instead of w* stride
    - cached _max_x with wrong dimensions
6. He Initialisation was wrong
    - fan_in  = in_featueres[1:] resulted in fan_in = out_features for Linear Layer.
    - After the fix the early layers and logits become more stabilized for further training
7. Core Problem, the way im2col flattened the image was not compatible with the shape of the image the convolutional layer was expecting for the calculation of back prop.
    - i produced columns in the order (i, j, n), but reshaped as if they were in (n, i, j) order
    - This happened because in our original im2col:
    ```
    0 col_idx = 0
    1 for i in range(H_out):
    2     hs = i*stride
    3     for j in range(W_out):
    4         ws = j * stride
    5         patch = x_p[ :, :, hs:hs+kH, ws:ws+kW] #shape (N, C, kH, kW)
    6         cols[col_idx:col_idx+N] = patch.reshape(N, -1) # (N, C*kH*kW)
    7         col_idx+=N
    ```
    Ln 6 writes N spatial blocks (one for each image). With the loop, for each (i,j) position, N spatial blocks are written. so intuitively the index notation of the matrix this loop is representing is (i, j, n). i for the first loop, j for the second loop, n for the row in the patch corresponding to our image.
    - The fix is very simple. We make the loop correspond to the shape (n, i , j). Because this means the allocations will be done in this order.
    0 row = 0
    1 for n in range(N):
    2   for i in range(H_out):
    3       hs = i * stride
    4       for j in range(W_out):
    5           ws = j *stride
    6           patch = x_p[n, :, hs:hs+hH, ws:ws+hW]  #now the shape is (C, kH, kW)
    7           cols[row] = patch.reshape(-1) # (C*kH*kW)
    8           row+=1
    - With same logic applied to col2im, the main issue is fixed, and the image can freely flow through forward and backward layer of Convolution

# What each diagnostic test meant?
1. Local_input gradient check per layer:
    - For a layer f, construct a syntethic Loss = <f(x), dout>, then check on a random subset of indices k (for speed) if the dL/dx is equal to the limit definition of a gradient for f at x.
    - This proves our backward implements the right Jacobian w.r.t inputs.
    - Improvements to the test over time:
        - Compare only over tested indices, because we are only calculating the contributions the subset k. The rest are initialised as 0 and will mistakingly give large errors.
2. Paramter Gradient Check:
    - Same idea but pertrube gradient entries instead
    - Proves dL/dθ is good. Early on this test was passed so we were able to isolate dL/dx
3. Forward row-order sanity:
    -  Calculates y_cols = im2col(x) @ W_col.T (+b) and compares to y flattened y.transpose(0, 2,3,1).reshape(N*H_out*W_out, C_out)
    - This makes sure that the way we flatten y matches the order which our backward layer expects in 

4. dx path order sanity:
    - Build dcols = dout_2d @ W_col then dx_algebra = im2col(dcols, ...) and compare to dx from backward(dout).
    - Proves entire input gradient dout -> dcols -> dx is correct / incorrect.
5. Vector - Jacobian product finite difference:
    - Compare analytic <dx, v> to central difference  (L(x + εv) - L(x - εv))/ 2ε
    - Checks the whole input. Less sensitive to per-index noise.

6. Adjoint Property of im2col/col2im:
    - <im2col(x),R>=<x,col2im(R)>
    - Why: proves col2im is the adjoint of im2col (they’re true inverses under inner product), crucial for correct dx.


