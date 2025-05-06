from data import train_data, test_data
from preprocess import build_vocab, prepare_data
from rnn import SimpleRNN

max_len = 10
embedding_dim = 16
hidden_dim = 64

# Sadece train verisi ile vocab oluştur
word2idx = build_vocab(train_data.keys())

# Verileri hazırla
X_train, y_train = prepare_data(train_data, word2idx, max_len)
X_test, y_test = prepare_data(test_data, word2idx, max_len)

# Model oluştur
rnn = SimpleRNN(vocab_size=len(word2idx), embedding_dim=embedding_dim, hidden_dim=hidden_dim)

# Eğitim
for epoch in range(30):
    epoch_loss = 0
    for x, y in zip(X_train, y_train):
        loss = rnn.backward(x, y)
        epoch_loss += loss
    print(f"Epoch {epoch+1} - Loss: {epoch_loss / len(X_train):.4f}")

# Test
correct = 0
for x, y_true in zip(X_test, y_test):
    y_pred = rnn.forward(x)
    prediction = int(y_pred.item() >= 0.5)
    print(f"Sigmoid Output: {y_pred.item():.4f} | Prediction: {prediction} | True: {y_true}")
    if prediction == y_true:
        correct += 1

accuracy = correct / len(X_test)
print(f"Test Accuracy: {accuracy:.2%}")
