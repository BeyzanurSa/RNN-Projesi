from data import train_data, test_data
from preprocess import build_vocab, prepare_data
from rnn import SimpleRNN  
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

max_len = 10
embedding_dim = 16
hidden_dim = 32

# Seçenek: "xavier", "classic", "scaled"
init_method = "xavier"  
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
train_losses = []
train_accuracies = []

for epoch in range(1, 301):  # 300 epoch'a kadar
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
    train_losses.append(avg_loss)
    train_accuracies.append(avg_acc)

    if epoch % 10 == 0:
        print(f"[Epoch {epoch}] Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.2%} | Avg Sigmoid Output: {avg_output:.4f}")

    # Early stopping
    if epoch > 10 and abs(train_losses[-1] - train_losses[-2]) < 1e-5:
        print(f"Early stopping at epoch {epoch}")
        break

# Grafik: Loss ve Accuracy
# Loss grafiği
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(train_losses) + 1), train_losses, color='red')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Eğitim Kaybı (Loss)")
plt.grid(True)
plt.tight_layout()
plt.savefig('images/loss_rnn.png')
plt.show()

# Accuracy grafiği
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, color='blue')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Eğitim Doğruluğu (Accuracy)")
plt.grid(True)
plt.tight_layout()
plt.savefig('images/accuracy_rnn.png')
plt.show()

# Test
correct = 0
y_true_list = []
y_pred_list = []

for x, y_true in zip(X_test, y_test):
    y_pred = rnn.forward(x)
    prediction = int(y_pred.item() >= 0.5)

    # Kelimeleri gösterme
    input_words = []
    for idx in x:
        if idx != 0:
            for word, word_idx in word2idx.items():
                if word_idx == idx:
                    input_words.append(word)
                    break

    print(f"Input: {' '.join(input_words)} | Sigmoid Output: {y_pred.item():.4f} | Prediction: {prediction} | True: {y_true}")
    
    if prediction == y_true:
        correct += 1

    y_true_list.append(y_true)
    y_pred_list.append(prediction)

accuracy = correct / len(X_test)
print(f"Test Accuracy: {accuracy:.2%}")

# Confusion Matrix
cm = confusion_matrix(y_true_list, y_pred_list)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negatif", "Pozitif"])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.grid(False)
plt.savefig('images/confusion_matrix_rnn.png')
plt.show()
