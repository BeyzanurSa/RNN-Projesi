import numpy as np

class SimpleRNN:
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Daha iyi başlangıç ağırlıkları (Xavier/Glorot initialization)
        self.embedding = np.random.randn(vocab_size, embedding_dim) / np.sqrt(vocab_size)
        self.Wxh = np.random.randn(embedding_dim, hidden_dim) / np.sqrt(embedding_dim)
        self.Whh = np.random.randn(hidden_dim, hidden_dim) / np.sqrt(hidden_dim)
        self.Why = np.random.randn(hidden_dim, 1) / np.sqrt(hidden_dim)
        self.bh = np.zeros((1, hidden_dim))
        self.by = np.zeros((1, 1))

    def sigmoid(self, x):
        # Numerik stabilite için clipping ekledik
        return 1 / (1 + np.exp(-np.clip(x, -15, 15)))

    def forward(self, x):
        h = np.zeros((1, self.hidden_dim))
        self.last_inputs = []
        self.last_hs = { -1: h }

        for t, word_idx in enumerate(x):
            word_embed = self.embedding[word_idx].reshape(1, -1)
            self.last_inputs.append(word_embed)
            h = np.tanh(word_embed @ self.Wxh + h @ self.Whh + self.bh)
            self.last_hs[t] = h

        y = self.sigmoid(h @ self.Why + self.by)
        return y

    def binary_cross_entropy(self, prediction, label):
        # Numerik stabilite için değerler sınırlandırıldı
        prediction = np.clip(prediction, 1e-7, 1 - 1e-7)
        return -(label * np.log(prediction) + (1 - label) * np.log(1 - prediction))

    def backward(self, x, label, learning_rate=0.05):
        # Öğrenme oranını artırdık
        y_pred = self.forward(x)
        loss = self.binary_cross_entropy(y_pred, label)

        d_y = y_pred - label
        d_Why = self.last_hs[len(x) - 1].T @ d_y
        d_by = d_y

        d_Whh = np.zeros_like(self.Whh)
        d_Wxh = np.zeros_like(self.Wxh)
        d_embedding = np.zeros_like(self.embedding)
        d_h_next = np.zeros((1, self.hidden_dim))

        for t in reversed(range(len(x))):
            h = self.last_hs[t]
            h_prev = self.last_hs[t - 1]

            # Gradyan akışı için Why'den gelen gradyan daha büyük
            if t == len(x) - 1:
                d_h = d_y @ self.Why.T * (1 - h * h)
            else:
                d_h = (d_h_next @ self.Whh.T) * (1 - h * h)
                
            d_Whh += h_prev.T @ d_h
            d_Wxh += self.last_inputs[t].T @ d_h

            if x[t] != 0:  # <PAD> token'ı güncelleme dışı bırakıldı
                d_embedding[x[t]] += (d_h @ self.Wxh.T).flatten()

            d_h_next = d_h

        # Gradyan patlama sorununu önlemek için clipping
        for grad in [d_Why, d_Whh, d_Wxh, d_embedding]:
            np.clip(grad, -5, 5, out=grad)

        # Güncelleme
        self.Why -= learning_rate * d_Why
        self.by -= learning_rate * d_by
        self.Whh -= learning_rate * d_Whh
        self.Wxh -= learning_rate * d_Wxh
        self.embedding -= learning_rate * d_embedding

        return loss.item()
