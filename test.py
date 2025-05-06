from data import test_data
from preprocess import build_vocab, prepare_data
from rnn import SimpleRNN


# Ayarlar
max_len = 10
embedding_dim = 16
hidden_dim = 64

# Aynı vocab kullanılmalı (eğitimdekiyle aynı sıralama)
word2idx = build_vocab(test_data.keys())

# Test verisini hazırla
X_test, y_test = prepare_data(test_data, word2idx, max_len)

# Eğitilmiş model yüklenmeli (biz basitçe yeniden eğitiyoruz örnek olsun diye)
rnn = SimpleRNN(vocab_size=len(word2idx), embedding_dim=embedding_dim, hidden_dim=hidden_dim)

# Modeli eğit → Not: Gerçek senaryoda eğitilen model kaydedilip buraya yüklenmeli
from data import train_data
X_train, y_train = prepare_data(train_data, word2idx, max_len)
for epoch in range(30):
    epoch_loss = 0
    for x, y in zip(X_train, y_train):
        loss = rnn.backward(x, y)
        epoch_loss += loss
    print(f"Epoch {epoch+1} - Loss: {epoch_loss / len(X_train):.4f}")


# Tahmin ve accuracy
correct = 0
for x, y_true in zip(X_test, y_test):
    y_pred = rnn.forward(x)
    print(f"Sigmoid Output: {y_pred.item():.4f}")
    prediction = int(y_pred.item() >= 0.5)
    print(f"Input: {x} | Prediction: {prediction} | True: {y_true}")
    if prediction == int(y_true):
        correct += 1


accuracy = correct / len(y_test)
print(f"Test Accuracy: {accuracy:.2%}")
