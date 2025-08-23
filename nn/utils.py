from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
def he_init(shape):
    # For ReLu nets W ~ N(0, sqrt(2/n))
    # weight_shape:
    #   For conv: (num_filters_out, num_channels_in, kernel_height, kernel_width)
    #   For linear: (num_inputs, num_outputs)
    if len(shape)== 2 : # Linear
        fan_in = shape[0]
    elif len(shape) == 4: #Conv
        fan_in = shape[1] * shape[2] * shape[3]
    else:
        fan_in = np.prod(shape[2:])
    variance = np.sqrt(2.0 / fan_in)
    return np.random.randn(*shape).astype(np.float32) * variance # rand expects argument telling it in which dimension to return the number
    

def im2col(x, kH, kW, stride=1, pad=0):
    # x: (N, C, H, W)  ->  cols: (N*H_out*W_out, C*kH*kW) in (n,i,j) row order
    N, C, H, W = x.shape
    H_out = (H + 2*pad - kH)//stride + 1
    W_out = (W + 2*pad - kW)//stride + 1

    x_p = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)), mode='constant')
    cols = np.zeros((N*H_out*W_out, C*kH*kW), dtype=x.dtype)

    row = 0
    for n in range(N):
        for i in range(H_out):
            hs = i * stride
            for j in range(W_out):
                ws = j * stride
                patch = x_p[n, :, hs:hs+kH, ws:ws+kW]              # (C,kH,kW)
                cols[row] = patch.reshape(-1)                      # (C*kH*kW,)
                row += 1
    return cols, H_out, W_out

def col2im(cols, x_shape, kH, kW, stride=1, pad=0, H_out=None, W_out=None):
    # inverse of the im2col above (same (n,i,j) row order)
    N, C, H, W = x_shape
    if H_out is None:
        H_out = (H + 2*pad - kH)//stride + 1
    if W_out is None:
        W_out = (W + 2*pad - kW)//stride + 1

    dx_p = np.zeros((N, C, H + 2*pad, W + 2*pad), dtype=cols.dtype)

    row = 0
    for n in range(N):
        for i in range(H_out):
            hs = i * stride
            for j in range(W_out):
                ws = j * stride
                patch = cols[row].reshape(C, kH, kW)
                dx_p[n, :, hs:hs+kH, ws:ws+kW] += patch
                row += 1

    return dx_p[:, :, pad:pad+H, pad:pad+W]

def step_lr(epoch, base_lr, milestones=(30, 60), gamma=0.1):
    m = sum(epoch >= e for e in milestones)  # how many milestones passed
    return base_lr * (gamma ** m)

def cosine_lr(epoch, total_epochs, base_lr, min_lr=0.0, warmup=5):
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    t = (epoch - warmup) / max(1, (total_epochs - warmup))
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * t))