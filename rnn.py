import numpy as np

class SimpleRNN:
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Ağırlıklar
        self.embedding = np.random.randn(vocab_size, embedding_dim) * 0.1
        self.Wxh = np.random.randn(embedding_dim, hidden_dim) * 0.1
        self.Whh = np.random.randn(hidden_dim, hidden_dim) * 0.1
        self.Why = np.random.randn(hidden_dim, 1) * 0.1
        self.bh = np.zeros((1, hidden_dim))
        self.by = np.zeros((1, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        # x: (seq_len,)
        h = np.zeros((1, self.hidden_dim))
        self.last_inputs = []
        self.last_hs = { -1: h }

        for t, word_idx in enumerate(x):
            word_embed = self.embedding[word_idx].reshape(1, -1)  # (1, embedding_dim)
            self.last_inputs.append(word_embed)
            h = np.tanh(word_embed @ self.Wxh + h @ self.Whh + self.bh)
            self.last_hs[t] = h

        y = self.sigmoid(h @ self.Why + self.by)  # (1, 1)
        return y

    def binary_cross_entropy(self, prediction, label):
        return -(label * np.log(prediction + 1e-8) + (1 - label) * np.log(1 - prediction + 1e-8))

    def backward(self, x, label, learning_rate=0.1):
        y_pred = self.forward(x)
        loss = self.binary_cross_entropy(y_pred, label)

        # dL/dy
        d_y = y_pred - label  # derivative of BCE w.r.t output

        # dL/dWhy
        d_Why = self.last_hs[len(x) - 1].T @ d_y
        d_by = d_y

        # Backprop through time
        d_Whh = np.zeros_like(self.Whh)
        d_Wxh = np.zeros_like(self.Wxh)
        d_embedding = np.zeros_like(self.embedding)
        d_h_next = np.zeros((1, self.hidden_dim))

        for t in reversed(range(len(x))):
            h = self.last_hs[t]
            h_prev = self.last_hs[t - 1]

            # Backprop through tanh
            d_h = (d_y @ self.Why.T + d_h_next) * (1 - h ** 2)
            d_Whh += h_prev.T @ d_h
            d_Wxh += self.last_inputs[t].T @ d_h
            d_embedding[x[t]] += (d_h @ self.Wxh.T).flatten()
            d_h_next = d_h @ self.Whh.T

        # Güncelle
        self.Why -= learning_rate * d_Why
        self.by -= learning_rate * d_by
        self.Whh -= learning_rate * d_Whh
        self.Wxh -= learning_rate * d_Wxh
        self.embedding -= learning_rate * d_embedding

        return loss.item()
