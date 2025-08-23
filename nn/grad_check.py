import numpy as np

def _gather_layer_inputs(model, x0):
    acts = [x0]
    a = x0
    for layer in model.layers:
        a = layer.forward(a)
        acts.append(a)
    return acts  # len = L+1

def _is_nondiff(layer):
    from types import SimpleNamespace
    # Adjust class names if needed
    return layer.__class__.__name__ in ("ReLU", "MaxPool2D")

def check_input_grads_layer(layer, x_in, eps=1e-4, atol=1e-2, rtol=1e-2,
                            seed=0, max_checks=128, jitter_non_smooth=True, jitter=1e-6,
                            skip_ties_for_pool=True, pool_h=None, pool_w=None, stride=None):
    rng = np.random.default_rng(seed)
    x = x_in.copy()
    # Jitter to break ties ONLY for non-smooth layers
    if jitter_non_smooth and _is_nondiff(layer):
        x += jitter * rng.standard_normal(x.shape).astype(x.dtype)

    out = layer.forward(x)
    dout = rng.standard_normal(out.shape).astype(out.dtype)
    dx = layer.backward(dout)

    # Optional: for MaxPool, skip inputs whose pooling window has ties
    mask = None
    if skip_ties_for_pool and layer.__class__.__name__ == "MaxPool2D":
        ph, pw, s = layer.pool_h, layer.pool_w, layer.stride
        N, C, H, W = x.shape
        H_out = (H - ph)//s + 1
        W_out = (W - pw)//s + 1
        safe = np.zeros_like(x, dtype=bool)
        for i in range(H_out):
            hs = i*s
            for j in range(W_out):
                ws = j*s
                win = x[:, :, hs:hs+ph, ws:ws+pw]           # (N,C,ph,pw)
                flat = win.reshape(N, C, ph*pw)
                maxv = flat.max(axis=2, keepdims=True)
                ties = np.sum(flat == maxv, axis=2)         # (N,C)
                # mark only the unique max position as safe
                # (safe mask used only to compute max error on checked entries)
                arg = flat.argmax(axis=2)
                y = hs + (arg // pw)
                z = ws + (arg %  pw)
                unique = (ties == 1)
                # write safe at those unique max coords
                idxN = np.arange(N)[:,None]
                idxC = np.arange(C)[None,:]
                safe[idxN, idxC, y, z] = unique
        mask = safe  # only compare these entries

    dx_num = np.zeros_like(x)
    # subsample indices for speed
    flat_idx = np.arange(x.size)
    if max_checks is not None and max_checks < x.size:
        if mask is not None:
            okpos = np.flatnonzero(mask.reshape(-1))
            if okpos.size:
                flat_idx = rng.choice(okpos, size=min(max_checks, okpos.size), replace=False)
            else:
                flat_idx = rng.choice(flat_idx, size=max_checks, replace=False)
        else:
            flat_idx = rng.choice(flat_idx, size=max_checks, replace=False)

    for k in flat_idx:
        idx = np.unravel_index(k, x.shape)
        old = x[idx]
        x[idx] = old + eps; Lpos = np.sum(layer.forward(x) * dout)
        x[idx] = old - eps; Lneg = np.sum(layer.forward(x) * dout)
        x[idx] = old
        dx_num[idx] = (Lpos - Lneg) / (2*eps)
        # --- compare only on tested indices (and intersect with pool's unique-max mask) ---
    tested = np.zeros_like(dx, dtype=bool).reshape(-1)
    tested[flat_idx] = True
    tested = tested.reshape(dx.shape)

    if mask is not None:          # for MaxPool, only compare at unique-max positions
        tested &= mask
        if not np.any(tested):
            return True, 0.0

    ok = np.allclose(dx[tested], dx_num[tested], atol=atol, rtol=rtol)
    max_err = float(np.max(np.abs(dx[tested] - dx_num[tested]))) if np.any(tested) else 0.0
    return ok, max_err


def check_param_grads_layer(layer, x_in, eps=1e-4, atol=1e-2, rtol=1e-2, seed=0, max_checks=64):
    if not hasattr(layer, "params") or not layer.params():
        return []
    rng = np.random.default_rng(seed)
    out = layer.forward(x_in)
    dout = rng.standard_normal(out.shape).astype(out.dtype)
    layer.backward(dout)  # fills grads

    results = []
    for i, (W, dW) in enumerate(layer.params()):
        total = W.size
        num = min(max_checks, total)
        flat_idx = rng.choice(np.arange(total), size=num, replace=False)
        dW_num = np.zeros_like(W)
        for k in flat_idx:
            idx = np.unravel_index(k, W.shape)
            old = W[idx]
            W[idx] = old + eps; Lpos = np.sum(layer.forward(x_in) * dout)
            W[idx] = old - eps; Lneg = np.sum(layer.forward(x_in) * dout)
            W[idx] = old
            dW_num[idx] = (Lpos - Lneg) / (2*eps)
        tested = np.zeros_like(W, dtype=bool).reshape(-1); tested[flat_idx]=True; tested=tested.reshape(W.shape)
        ok = np.allclose(dW[tested], dW_num[tested], atol=atol, rtol=rtol)
        max_err = float(np.max(np.abs(dW[tested]-dW_num[tested])))
        results.append((f"param[{i}]", ok, max_err))
    return results

def check_model_grads(model, x0, eps=1e-4, atol=1e-2, rtol=1e-2, seed=0, verbose=True):
    inputs = _gather_layer_inputs(model, x0)
    in_reports, pa_reports = [], []
    for li, layer in enumerate(model.layers):
        name = layer.__class__.__name__
        x_in = inputs[li]
        ok_in, max_in = check_input_grads_layer(
            layer, x_in, eps=eps, atol=atol, rtol=rtol, seed=seed,
            max_checks=128, jitter_non_smooth=True, skip_ties_for_pool=True
        )
        in_reports.append((name, ok_in, max_in))
        if verbose: print(f"[IN ] {li:02d} {name:18s} ok={ok_in} max_err={max_in:.2e}")
        pres = check_param_grads_layer(layer, x_in, eps=eps, atol=atol, rtol=rtol, seed=seed, max_checks=32)
        pa_reports.append((name, pres))
        if verbose:
            for (pname, okp, mx) in pres:
                print(f"[PAR] {li:02d} {name:18s} {pname:8s} ok={okp} max_err={mx:.2e}")
    return in_reports, pa_reports
