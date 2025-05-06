from data import train_data, test_data
from preprocess import build_vocab, prepare_data
from rnn import SimpleRNN  # Düzeltilmiş RNN sınıfını kullan

max_len = 10
embedding_dim = 16  # Boyutu artırdık
hidden_dim = 32   # Boyutu artırdık

# Sadece train verisi ile vocab oluştur
word2idx = build_vocab(train_data.keys())

# Verileri hazırla
X_train, y_train = prepare_data(train_data, word2idx, max_len)
X_test, y_test = prepare_data(test_data, word2idx, max_len)

# Model oluştur
rnn = SimpleRNN(vocab_size=len(word2idx), embedding_dim=embedding_dim, hidden_dim=hidden_dim)

# Eğitim
losses = []
for epoch in range(30):  # Epoch sayısını artırdık
    epoch_loss = 0
    for x, y in zip(X_train, y_train):
        loss = rnn.backward(x, y)
        epoch_loss += loss
    avg_loss = epoch_loss / len(X_train)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")
    
    # Early stopping kontrolü
    if epoch > 5 and abs(losses[-1] - losses[-2]) < 1e-5:
        print(f"Early stopping at epoch {epoch+1}")
        break

# Test
correct = 0
for x, y_true in zip(X_test, y_test):
    y_pred = rnn.forward(x)
    prediction = int(y_pred.item() >= 0.5)
    # Girdileri kelime olarak göster
    input_words = []
    for idx in x:
        if idx != 0:  # PAD token'ı gösterme
            for word, word_idx in word2idx.items():
                if word_idx == idx:
                    input_words.append(word)
                    break

    print(f"Input: {' '.join(input_words)} | Sigmoid Output: {y_pred.item():.4f} | Prediction: {prediction} | True: {y_true}")
    if prediction == y_true:  # Girinti düzeltildi
        correct += 1

accuracy = correct / len(X_test)
print(f"Test Accuracy: {accuracy:.2%}")

