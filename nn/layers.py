import numpy as np
from utils import col2im, im2col, he_init


class Conv2D:
    # Convolutional layer from scratch
    # Weights W shape (C_in, C_out, kH, kW)
    # Biases b shape (C_out)
    #Forward: im2col(x) -> @ W_cols -> reshape to (N, C_out, H_out, W_out)
    #Backward: 
    #   dx = col2im(dout * W_col, x.shape, kH, kW, stride, pad)
    #   dW = (dout.T @ cols).reshape(W.shape)
    #   db = sum(dout, axis= 0)

    def __init__(self, C_in, C_out, kH, kW, stride=1, pad= 0, bias =True, weight_scale = None):
        self.stride = int(stride)
        self.pad = int(pad)
        #init weights
        if weight_scale is None:
            self.W = he_init((C_out, C_in, kH, kW))
        else:
            self.W = (np.random.randn(C_out, C_in, kH, kW).astype(np.float32) * float(weight_scale))
        self.b = np.zeros((C_out,),dtype=np.float32) if bias else None
        
        #grads

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        # cache placeholder
        self._x_shape = None
        self._cols = None
        self._H_out = None
        self._W_out = None # cached output width
        self._W_col = None # cached flattened weights

        
    def forward(self, x):
        # x (N, C, H, W) -> (N, C_out, H_out, W_out)
        self._x_shape = x.shape
        C_out, C_in, kH, kW = self.W.shape

        # 1. extract patches
        cols, H_out, W_out = im2col(x, kH, kW, stride = self.stride, pad = self.pad) # (N*H_out*W_out, C_in*kH*kW)
        
        # 2. flatten weights 
        W_col = self.W.reshape(C_out, -1) # (C_out, C_in * kH *kW)

        # 3. multiply patches with eights
        out_cols = cols @ W_col.T # (N*H_out*W_out, C_in*kH*kW) @ (C_in * kH *kW, C_out) = (N*H_out*W_out, C_out)
        if self.b is not None:
            out_cols += self.b
        
        # 4. Reshape to (N, C_out, C_in, H_out, W_out)
        N = x.shape[0] #(N, C, H, W)[0] = N
        out = out_cols.reshape(N, H_out, W_out, C_out).transpose(0,3,1,2) #(N*H_out*W_out, C_out).reshape(N, H_out, W_out, C_out) => (N, H_out, W_out, C_out).transpose(0,3,1,2) = (N, C_out, H_out, W_out)

        #5. cache for backward
        self._cols = cols
        self._H_out = H_out
        self._W_out = W_col
        self._W_col = W_col

        return out 
    
    def backward(self, dout):
        # dout (N, C_out, H_out, W_out)
        # returns dx (N, C_in, H, W)
        #also fills self.dW self.db
        N = self._x_shape[0] #(N, C, H, W)[0] = N
        C_out, C_in, kH, kW = self.W.shape

        # match forward out_cols layout (N*H_out*W_out, C_out)
        dout_2d = dout.transpose(0, 3, 1,2).reshape(-1, C_out)

        # find db: sum over all positions and batch
        if self.b is not None:
            self.db = dout_2d.sum(axis = 0).astype(np.float32)
        # find dW: dout_2d @ cols (x_2d)
        dW_col = dout_2d @ self._cols #(N*H_out*W_out, C_out).T @ (N*H_out*W_out, C_in * kH + kW) = (C_out,C_in * kH + kW)
        self.dW = dW_col.reshape(self.W.shape).astype(np.float32)

        # dx 
        dcols = self._cols @ dW_col
        dx = col2im(dcols, x_shape = self._x_shape, kH=kH, kW= kW, stride= self.stride, pad= self.pad, H_out= self._H_out, W_out= self._W_out)

        return dx.astype(np.float32)






