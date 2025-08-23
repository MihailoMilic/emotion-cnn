import numpy as np

class SGD:
    """
    Stochastic Gradient Descent with optional momentum and Nesterov.
    - momentum: 0.0 disables momentum (pure SGD)
    - nesterov: True uses Nesterov accelerated gradient (requires momentum > 0)
    - weight_decay: L2 regularization (like PyTorch's SGD when decoupled_wd=False)
    - decoupled_wd: if True, applies AdamW-style decoupled weight decay
    - dampening: rarely needed; use 0.0 for standard momentum
    """
    def __init__(self, lr=1e-1, weight_decay=0.0, momentum=0.0, nesterov=False,
                 dampening=0.0, decoupled_wd=False):
        self.lr = float(lr)
        self.wd = float(weight_decay)
        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)
        self.dampening = float(dampening)
        self.decoupled_wd = bool(decoupled_wd)

        if self.nesterov and self.momentum <= 0:
            raise ValueError("Nesterov requires momentum > 0")

        # Per-parameter velocity buffers, keyed by id(param)
        self._velocity = {}

    def _get_velocity(self, param):
        pid = id(param)
        v = self._velocity.get(pid, None)
        if v  is None or v.shape != param.shape:
            v = np.zeros_like(param)
            self._velocity[pid] = v
        return v

    def  step(self, model):
        for param, grad in model.params():
            if grad is None:
                continue
            d_p =  grad
            if self.wd and not self.decoupled_wd and param.ndim > 1:
                d_p = d_p + self.wd * param

            # ---- Momentum
            if self.momentum > 0.0:
                v = self._get_velocity(param )

                v *= self.momentum
                v += (1.0 - self.dampening) * d_p

                if self.nesterov :
                    step_dir = d_p + self.momentum * v
                else:
                    step_dir = v
            else:
                step_dir = d_p
            if self.decoupled_wd and self.wd and param.ndim > 1:
                param *= (1.0 - self.lr * self.wd)


            param -= self.lr * step_dir
