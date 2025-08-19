import numpy as np
from .utils import col2im, im2col, he_init


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
        self._W_out = W_out
        self._W_col = W_col

        return out 
    
    def backward(self, dout):
        # dout (N, C_out, H_out, W_out)
        # returns dx (N, C_in, H, W)
        #also fills self.dW self.db

        N = self._x_shape[0] #(N, C, H, W)[0] = N
        C_in, _, kH, kW = self.W.shape
        _, C_out, H_out, W_out = dout.shape
        
        # match forward out_cols layout (N*H_out*W_out, C_out)
        dout_2d = dout.transpose(0, 2,3,1).reshape(N*H_out*W_out, C_out)

        # find db: sum over all positions and batch
        if self.b is not None:
            self.db = dout_2d.sum(axis = 0).astype(np.float32)
        # find dW: dout_2d @ cols (x_2d)
        dW_col = dout_2d.T @ self._cols #(N*H_out*W_out, C_out).T @ (N*H_out*W_out, C_in * kH + kW) = (C_out,C_in * kH + kW)
        self.dW = dW_col.reshape(self.W.shape).astype(np.float32)

        # dx 
        dcols = dout_2d @ self._W_col # (N*H_out*W_out, C_out) @ (C_out, C_in * kH *kW) = (N*H_out*W_out, C_in * kH *kW))

        dx = col2im(dcols, x_shape = self._x_shape, kH=kH, kW= kW, stride= self.stride, pad= self.pad, H_out= self._H_out, W_out= self._W_out)

        return dx.astype(np.float32)
    def params(self):
        params = [(self.W, self.dW)]
        if self.b is not None:
            params.extend([(self.b, self.db)])
        return params


class ReLU:
    # Element-wise ReLU transformation
    # forward: out = max(0, x)
    # backward: dx = dout * (x>0)
    def __init__(self):
        self.mask = None
    def forward(self, x):
        self.mask = (x > 0) # (x > 0) is an array of same shape as x. mask_i = 1 if x_i > 0 else 0. 
        return x * self.mask # in numpy * multiplies arrays element wise. Result is new x, where all negative terms are turned into a zero. 

    def backward(self, dout):
        return dout * self.mask # result is that we only care about gradients which came from non-zero entries of x 
    def params(self):
        return []

class MaxPool2D:
    # 2D pooling (channel first)
    #pool: (pool_h, pool_w)
    # stride: step between pooling windows
    #Stores argmax positions to route gradients back, meaning we only care about gradients in argmax positions
    def __init__(self, pool_w =2, pool_h = 2, stride = 2):
        self.pool_w = int(pool_w)
        self.pool_h =int( pool_h)
        self.stride = int(stride)

        self._x_shape = None
        #argmax indices
        self._max_y = None 
        self._max_x = None

    def forward(self, x):
        N, C, H, W = x.shape
        ph, pw, s = self.pool_h, self.pool_w, self.stride
        H_out = (H-ph) // s +1 # very similar patch-dimension-computation to the one we did for im2col
        W_out = (W- pw) //s +1

        out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
        
        self._max_y = np.zeros((N, C, H_out, W_out), dtype = np.int32)
        self._max_x = np.zeros((N,C,H,W), dtype= np.int32)

        for i in range(H_out):
            hs = i *self.stride
            for j in range(W_out):
                ws = j + self.stride
                window = x[: , : , hs:hs+ph, ws:ws+pw] # (N, C, ph, pw)
                flat = window.reshape(N, C, -1) # (N, C, ph*pw)
                arg = flat.argmax(axis =2) # (N, C)  finds the index of the largest in each array of window's pixels, there is N*C of these indices
                # [[ 0.2, 0.5, 0.1, 0.3]]    argmax   [[1]]
                # [[ 0.8, 0.4, 0.6, 0.7]]      ⟹     [[0]]
                # [[ 0.1, 0.2, 0.9, 0.4]]             [[2]]                                              numpy broadcasting turns all arrays into the same shape if they have one axis of same dimension like (N,1) (1,C)
                out[: , :, i, j ] = flat[np.arange(N)[:, None], np.arange(C)[None, :], arg] # (N, 1), (1, C), (N, C) => (N,C) (N,C) (N,C)
                #out is of shape N, C, H_out, W_out, meaning that every [:,:, i,j] entry corresponds to a single pooling window of x. What the loop does is finds the window for Ɐ n, c in N, C, finds the biggest entry for Ɐ n, c in N, C, and then places them in out[:, :, i, j], meaning that we simply record the previous result. On top of that due to the nature of our "record" we are able to iteratively go over all i and j and record them all in one big tensor. that is everything that happens here
                # N = 2, C = 1
                #args  = [[1],[3]]
                #np.arange(2)[:, None] = [[0], [1]] (2,1)
                #np.arange(1)[:, None] = [[0]] (1,1)    
                # broadcast 
                #    ↓
                # [[0],[1]] and [[0], [0]], and [[1], [3]] so if we index over all of them we get, (0,0,1) and (1,0,3), effectively indexing over together.
                # so in our example we do what we do in flat so that we index over range of N and C together and over each arg which corresponds to these n and c in N, C.
                # so at the end flat[np.arange(N)[:, None], np.arange(C)[None, :], arg] is an array (N,C) where at each entry is the largest entry in flat[n, c, :]
                
                y_idx = arg // pw  # find the height index by floor-dividing the index in the argument with number of rows
                x_idx = arg % pw  # find the row index by finding the remainder
                # (y_idx, x_idx) are coordinates with respect to the pooling window
                # to find the coordinates in the original input tesnor we add to them hs and ws
                self._max_y[:, :, i, j] = hs + y_idx 
                self._max_x[:, :, i, j] = ws + x_idx

        self._x_shape = x.shape
        return out

    def backward(self, dout):
        
        # dout: (N, C, H_out, W_out) -> dx: (N, C, H, W)
    
        N, C, H, W = self._x_shape
        dx = np.zeros((N, C, H, W), dtype=dout.dtype)
        H_out, W_out = dout.shape[2], dout.shape[3]

        #scatter add
        for i in range(H_out):
            for j in range(W_out):
                ys = self._max_y[:, :, i, j]   # (N, C)
                xs = self._max_x[:, :, i, j]   # (N, C)
                # place dout[n,c,i,j] at (ys[n,c], xs[n,c])
                for n in range(N):
                    # on LHS for specific picture, get all channels. dx[scalar, (C,), (C,), (C,)] shape so numpy iterates over all of them at the same time. which in this case results in 
                    dx[n, np.arange(C), ys[n], xs[n]] += dout[n, :, i, j]
        return dx
    def params(self):
        return []


class Flatten:
    """
    Flattens (N, C, H, W) -> (N, D). Stores original shape to unflatten in backward.
    """
    def __init__(self):
        self._orig = None

    def forward(self, x):
        self._orig = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self._orig)
    def params(self):
        return []


class Linear:

    def __init__(self, in_features, out_features, bias=True, weight_scale=None):
        if weight_scale is None:
            self.W = he_init((in_features, out_features))  # He init suits ReLU downstream
        else:
            self.W = (np.random.randn(in_features, out_features).astype(np.float32) * float(weight_scale))
        self.b = np.zeros((out_features,), dtype=np.float32) if bias else None
        # grads
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b) if bias else None
        # cache
        self._x = None

    def forward(self, x):
        """
        x: (N, D) -> out: (N, out_features)
        """

        self._x = x
        out =  x  @ self.W 
        if self.b is not None:
            out = out + self.b
        return out

    def backward(self, dout):
        """
        dout: (N, out_features)
        returns: dx (N, D)
        also fills self.dW, self.db
        """
        self.dW = (self._x.T @ dout).astype(np.float32)
        if self.b is not None:
            self.db = dout.sum(axis=0).astype(np.float32)
        dx = (dout @ self.W.T).astype(np.float32)
        return dx
    def params(self):
        params = [(self.W, self.dW)]
        if self.b is not None:
            params.extend([(self.b, self.db)])
        return params