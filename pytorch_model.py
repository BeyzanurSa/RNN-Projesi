import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from data import train_data, test_data
from preprocess import build_vocab, prepare_data

class PyTorchRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PyTorchRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x'in boyutu: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
        
        # RNN katmanı için hidden state başlatma
        h0 = torch.zeros(1, x.size(0), self.rnn.hidden_size).to(x.device)
        
        # RNN ileri yayılım
        output, hidden = self.rnn(embedded, h0)  # output: [batch_size, seq_len, hidden_dim]
        
        # Sadece son zaman adımındaki çıktıyı kullanma
        output = output[:, -1, :]  # [batch_size, hidden_dim]
        
        # Tam bağlantılı katman ve sigmoid aktivasyonu
        output = self.fc(output)  # [batch_size, 1]
        output = self.sigmoid(output)
        
        return output

def train_and_evaluate():
    # Hyperparameters
    max_len = 10
    embedding_dim = 16
    hidden_dim = 32
    learning_rate = 0.01
    num_epochs = 300
    
    # Vocab oluşturma
    word2idx = build_vocab(train_data.keys())
    
    # Veri hazırlama
    X_train, y_train = prepare_data(train_data, word2idx, max_len)
    X_test, y_test = prepare_data(test_data, word2idx, max_len)
    
    # PyTorch tensörlerine dönüştürme
    X_train_tensor = torch.tensor(X_train, dtype=torch.long)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float).reshape(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float).reshape(-1, 1)
    
    # Model, kayıp fonksiyonu ve optimize edici tanımlama
    model = PyTorchRNN(vocab_size=len(word2idx), embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Eğitim için metrikleri saklama
    train_losses = []
    train_accuracies = []
    
    # Early stopping için değişkenler
    best_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    # Eğitim döngüsü
    for epoch in range(1, num_epochs + 1):
        # Eğitim moduna geç
        model.train()
        
        # İleri yayılım
        y_pred = model(X_train_tensor)
        loss = criterion(y_pred, y_train_tensor)
        
        # Geri yayılım
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Eğitim metriklerini hesaplama
        with torch.no_grad():
            model.eval()
            train_preds = (model(X_train_tensor) >= 0.5).float()
            train_acc = (train_preds == y_train_tensor).float().mean()
            
            # Metrikleri kaydetme
            current_loss = loss.item()
            train_losses.append(current_loss)
            train_accuracies.append(train_acc.item())
            
            # Her 10 epokta bir bilgi yazdırma
            if epoch % 10 == 0:
                print(f"[Epoch {epoch}] Loss: {current_loss:.4f} | Accuracy: {train_acc.item():.2%}")
            
            # Early stopping kontrolü
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience and epoch > 50:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Eğitim grafiklerini çizme
    # Loss grafiği
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(train_losses) + 1), train_losses, color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Eğitim Kaybı (Loss) - PyTorch RNN")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('images/pytorch_loss.png')
    plt.show()
    
    # Accuracy grafiği
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, color='blue')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Eğitim Doğruluğu (Accuracy) - PyTorch RNN")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('images/pytorch_accuracy.png')
    plt.show()
    
    # Test değerlendirmesi
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        test_loss = criterion(y_pred, y_test_tensor).item()
        predictions = (y_pred >= 0.5).float()
        test_accuracy = (predictions == y_test_tensor).float().mean().item()
        
        # Test sonuçlarını yazdırma
        print(f"\nTest Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_accuracy:.2%}")
        
        # Her test örneği için sonuçları yazdırma
        y_true_list = y_test_tensor.numpy().flatten().tolist()
        y_pred_list = predictions.numpy().flatten().tolist()
        
        for i, (x, y_true, y_pred_val) in enumerate(zip(X_test, y_test, y_pred.numpy())):
            # Kelimeleri gösterme
            input_words = []
            for idx in x:
                if idx != 0:
                    for word, word_idx in word2idx.items():
                        if word_idx == idx:
                            input_words.append(word)
                            break
            
            pred_label = 1 if y_pred_val >= 0.5 else 0
            print(f"Input: {' '.join(input_words)} | Sigmoid Output: {y_pred_val[0]:.4f} | Prediction: {pred_label} | True: {y_true}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_true_list, y_pred_list)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negatif", "Pozitif"])
        disp.plot(cmap='Blues')
        plt.title("Confusion Matrix - PyTorch RNN")
        plt.grid(False)
        plt.savefig('images/pytorch_confusion_matrix.png')
        plt.show()

# Ana fonksiyonu çağır
if __name__ == "__main__":
    train_and_evaluate()
