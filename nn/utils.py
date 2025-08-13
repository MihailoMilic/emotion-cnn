import numpy as np

def He_Keiming_Init(shape):
    # For ReLu nets W ~ N(0, sqrt(2/n))
    # weight_shape:
    #   For conv: (num_filters_out, num_channels_in, kernel_height, kernel_width)
    #   For linear: (num_inputs, num_outputs)
    
    inputs_per_neutron = np.prod(shape[1:])
    variance = np.sqrt(2.0 / inputs_per_neutron)
    return np.random.randn(*shape).astype(np.float32) * variance # rand expects argument telling it in which dimension to return the number
    

def im2col(x, kH, kW, stride = 1, pad = 0):
    # x: (N, H, W)
    N, C, H, W = x.shape
    # we will now compute the shape of the output of our convulation between
    # Our img 6x6:     First 3x3 kernel:   2nd:         ....
    #     x x x x x x   [x x x] x x x   x [x x x] x x               
    #     x x x x x x   [x x x] x x x   x [x x x] x x
    #     x x x x x x   [x x x] x x x   x [x x x] x x         
    #     x x x x x x   x x x x x x     x [x x x] x x
    #     x x x x x x   x x x x x x     x [x x x] x x
    #     x x x x x x   x x x x x x     x [x x x] x x
    #  each step represents one entry of our output. The last step is on W - kW + 1, in our case 3. 
    # stride and pad are vars that allow us to control this iteration more precisely.

    
    H_out = (H + 2*pad - kH)//stride +1
    W_out = (W + 2*pad - kW)//stride +1
    # we wish to use matrix multiplications over loops due to their efficiency in numpy
    #we want y in shape (N, C_out, H_out, W_out)
    # Fully vectorized approach:
    # from numpy.lib.stride_tricks import sliding_window_view
    # def im2col_fast(x, kH, kW, stride=1, pad=0):
    #     # x: (N,C,H,W)
    #     N, C, H, W = x.shape
    #     H_out = (H + 2*pad - kH)//stride + 1
    #     W_out = (W + 2*pad - kW)//stride + 1

    #     x_p = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')
    #     # windows: (N, C, H_out', W_out', kH, kW) with stride=1 first
    #     win = sliding_window_view(x_p, window_shape=(kH, kW), axis=(2, 3))
    #     # apply stride by slicing the H_out'/W_out' axes
    #     win = win[:, :, ::stride, ::stride, :, :]          # (N, C, H_out, W_out, kH, kW)
    #     # reshape to (N*H_out*W_out, C*kH*kW)
    #     cols = win.transpose(0, 2, 3, 1, 4, 5).reshape(N*H_out*W_out, C*kH*kW)
    #     return cols, H_out, W_out
    # We will still use a few loops to dodge blackboxes and keep our mathematical understanding of the code
    x_p = np.pad(x, ((0,0), (0,0), (pad, pad), (pad, pad)), mode = "constant")
    # cols is like a spreadsheet co
    cols = np.zeros((N* H_out* W_out ,C*kH*kW), dtype=x.dtype)
    #col = (x_out shape, kernel shape)
    col_idx = 0
    #we are looping over the output 
    for i in range(H_out):
        hs = i*stride
        for j in range(W_out):
            ws = j * stride
            patch = x_p[ :, :, hs:hs+kH, ws:ws+kW] # take out a patch (kH,kW) at each position (hs, ws)
            cols[col_idx:col_idx+N] = patch.reshape(N, -1) # (N, C*kH*kW) both sides are. the ordering is a list of patches, of length N. patch is also a multidim list. with .reshape we flatten it. so patch is now just a list
            col_idx+=N
    # cols is a list of flattened patches (which is a list). which follows the position-first order.
    return cols, H_out, W_out

def col2im(cols, x_shape, kH, kW, stride= 1, pad = 0, H_out = None, W_out = None):

    N, C, H, W = x_shape

    if H_out is None:
        H_out = (H +2*pad -kH)//stride +1
    if W_out is None:
        W_out = (W + 2*pad -kW)// stride +1
    
    dx_p = np.zeros((N, C, H + 2*pad, W + 2*pad), dtype=cols.dtype)
    col_idx = 0
    for i in range(H_out):
        hs = i* stride
        for j in range(W_out):
            ws = j*stride
            patch = cols[col_idx:col_idx+N].reshape(N, C, kH, kW)
            dx_p[: , : , hs:hs+kH,ws: ws: ws+kW]+= patch
            col_idx += N
    return dx_p[:, :, pad:pad+H, pad:pad+W]
#unit test for back propagation
def grad_check(layer_forward, layer_backward, x, eps = 1e-3, atol = 1e-2, rtol = 1e-2):
    # layer_forward(x) -> (out, cache)
    # layer_backward(dout, cache) -> dx
    # out is output of the layer
    #dout is derivative of loss wrt out, we initialise it with a random variable for testing purposes
    #cache are important variables from layer_forward which back propagation needs.
 

    np.random.seed(0)
    out, cache = layer_forward(x)
    dout = np.random.randn(*out.shape).astype(np.float32)
    dx = layer_backward(dout, cache)
    #numeric part

    dx_num = np.zeros_like(x)  # create a zero matrix of the same shape as x to store a numerical gradient of each element of x
    it = np.nditer(x,flags=['multi_index'], op_flags=['readwrite']) #iterator that allows us to visit each element of x once, making it much simpler due to x's multidimensional index. (N,C,H,W)
    while not it.finished:
        idx = it.multi_index
        old = x[idx]
        x[idx] = old +eps
        out_pos, _ = layer_forward(x)
        x[idx] = old -eps
        out_neg, _ =layer_backward(x)
        x[idx] = old
        dx_num[idx] = ((out_pos-out_neg)*dout).sum() / (2*eps) # dL/dx[idx] = dL/dout * dout/dx[idx], which why we multiply by dout
        it.iternext()
    diff = np.abs(dx - dx_num)
    ok = np.allclose(dx, dx_num, atol=atol, rtol=rtol)
    return ok, diff.max(), (dx.mean(), dx.std()), (dx_num.mean(), dx_num.std())









