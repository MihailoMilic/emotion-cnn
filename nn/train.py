import numpy as np
import matplotlib.pyplot as plt


def iterate_minibatches(X, y, batch_size, shuffle = True):
    N = X.shape[0]
    idx = np.arange(N)
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, N - N % batch_size, batch_size):
        sl = idx[start: start+batch_size]
        yield X[sl], y[sl]


# We are feeding 
def evaluate(model, loss_fn, X, y, batch_size = 256):
    model.zero_grad()
    total_loss, total_correct, total = 0.0, 0.0, 0
    for Xb, yb in iterate_minibatches(X, y, batch_size, False):
        logits = model.forward()
        loss, dlogits = loss_fn(logits, yb)
        total_loss += loss * Xb.shape[0] # We multiply by the batch size because we are calculating the batch_size mean. Later we will divide by the total number of data examples to finish the mean function
        preds = np.argmax(logits, axis=1)
        total_correct += (preds == yb).sum()
        total += Xb.shape[0]
    return total_loss/ total, total_correct / total

def plot_epoch_losses(history_epoch_losses):
    num_epochs = len(history_epoch_losses)
    first, middle, last = 0, num_epochs //2 , num_epochs -1
    for e in sorted({first, middle, last}):
        y = np.asarray(history_epoch_losses[e], dtype=float)
        x = np.arange(1, len(y)+1)
        plt.plot(x,y, label = f'Epoch {e+1}')
        plt.xlabel("Iteration (batch)")
        plt.ylabel("Loss")
        plt.title("Training Loss per Iteration")
        plt.legend()
        plt.tight_layout()
        plt.show()

def train(model, loss_fn, optimizer, X_train, y_train, X_val = None, y_val = None, batch_size = 64, epochs = 10, print_every =1):
    N = X_train.shape[0]
    history_epoch_losses = []
    for epoch in range(epochs):
        epoch_losses = []
        model.zero_grad()
        running_loss, running_correct, seem = 0.0, 0.0,0
        for Xb, yb in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            logits = model.forward()
            loss, dlogits = loss_fn(logits, yb)
            epoch_losses.append(float(loss))
            model.backward()
            optimizer.step()
            model.zero_grad()

            preds = np.argmax(logits, axis =1)
            running_loss += loss * Xb.shape[0]
            running_correct += (preds == yb).sum()  # array of binary bools, summed to give number of correct answers
            seen += Xb.shape[0]
        
        train_loss = running_loss / seen
        train_correct= running_correct / seen
        history_epoch_losses.append(epoch_losses)
        if X_val is not None:
            val_loss, val_correct = evaluate(model, loss_fn, X_val, y_val)
            if epoch % print_every ==0:
                print(f"Epoch {epoch: 02d} | loss {train_loss:.4f} | acc {train_correct: .4f} | \n val_loss {val_loss: .4f} | correct {val_correct: .4f} ")
        else:
             if epoch % print_every ==0:
                print(f"Epoch {epoch: 02d} | loss {train_loss:.4f} | acc {train_correct: .4f} ")
    plot_epoch_losses(history_epoch_losses)


