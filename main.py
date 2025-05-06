from data import train_data, test_data
from preprocess import build_vocab, prepare_data
from rnn import SimpleRNN  # Düzeltilmiş RNN sınıfını kullan

max_len = 10
embedding_dim = 16
hidden_dim = 32

# Seçenek: "xavier", "classic", "scaled"
init_method = "xavier"  # "xavier" seçildiğinde geçerli
scale_value = 0.1  # scaled seçildiğinde geçerli



# Vocab oluştur
word2idx = build_vocab(train_data.keys())

# Verileri hazırla
X_train, y_train = prepare_data(train_data, word2idx, max_len)
X_test, y_test = prepare_data(test_data, word2idx, max_len)

# Model oluştur (init_method parametresiyle)
rnn = SimpleRNN(vocab_size=len(word2idx),
                embedding_dim=embedding_dim,
                hidden_dim=hidden_dim,
                init_method=init_method,
                scale=scale_value)
# Eğitim
losses = []
for epoch in range(1, 101):  # 100 epoch'a kadar uzattık, istediğinde azaltabilirsin
    epoch_loss = 0
    epoch_outputs = []
    epoch_correct = 0

    for x, y in zip(X_train, y_train):
        y_pred = rnn.forward(x)
        loss = rnn.backward(x, y)
        epoch_loss += loss

        prediction = int(y_pred.item() >= 0.5)
        if prediction == y:
            epoch_correct += 1
        epoch_outputs.append(y_pred.item())

    avg_loss = epoch_loss / len(X_train)
    avg_acc = epoch_correct / len(X_train)
    avg_output = sum(epoch_outputs) / len(epoch_outputs)
    losses.append(avg_loss)

    if epoch % 10 == 0:
        print(f"[Epoch {epoch}] Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.2%} | Avg Sigmoid Output: {avg_output:.4f}")

    # Early stopping
    if epoch > 10 and abs(losses[-1] - losses[-2]) < 1e-5:
        print(f"Early stopping at epoch {epoch}")
        break


# Test
correct = 0
for x, y_true in zip(X_test, y_test):
    y_pred = rnn.forward(x)
    prediction = int(y_pred.item() >= 0.5)

    input_words = []
    for idx in x:
        if idx != 0:
            for word, word_idx in word2idx.items():
                if word_idx == idx:
                    input_words.append(word)
                    break

    
    if prediction == y_true:
        correct += 1

accuracy = correct / len(X_test)
print(f"Test Accuracy: {accuracy:.2%}")
