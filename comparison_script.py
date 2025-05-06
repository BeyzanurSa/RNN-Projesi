import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Manuel oluşturulan RNN için
from data import train_data, test_data
from preprocess import build_vocab, prepare_data
from rnn import SimpleRNN

# PyTorch RNN için
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_model import PyTorchRNN  # Önceden oluşturduğumuz PyTorch modeli

def evaluate_custom_rnn():
    print("===== Manuel Oluşturulan SimpleRNN Modeli Değerlendirmesi =====")
    start_time = time.time()
    
    # Hyperparameters
    max_len = 10
    embedding_dim = 16
    hidden_dim = 32
    init_method = "xavier"
    
    # Vocab oluştur
    word2idx = build_vocab(train_data.keys())
    
    # Verileri hazırla
    X_train, y_train = prepare_data(train_data, word2idx, max_len)
    X_test, y_test = prepare_data(test_data, word2idx, max_len)
    
    # Model oluştur
    rnn = SimpleRNN(vocab_size=len(word2idx),
                    embedding_dim=embedding_dim,
                    hidden_dim=hidden_dim,
                    init_method=init_method)
    
    # Eğitim
    train_losses = []
    train_accuracies = []
    
    for epoch in range(1, 301):
        epoch_loss = 0
        epoch_correct = 0
        
        for x, y in zip(X_train, y_train):
            y_pred = rnn.forward(x)
            loss = rnn.backward(x, y)
            epoch_loss += loss
            
            prediction = int(y_pred.item() >= 0.5)
            if prediction == y:
                epoch_correct += 1
        
        avg_loss = epoch_loss / len(X_train)
        avg_acc = epoch_correct / len(X_train)
        train_losses.append(avg_loss)
        train_accuracies.append(avg_acc)
        
        if epoch % 50 == 0:
            print(f"[Epoch {epoch}] Loss: {avg_loss:.4f} | Accuracy: {avg_acc:.2%}")
        
        # Early stopping
        if epoch > 10 and abs(train_losses[-1] - train_losses[-2]) < 1e-5:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Eğitim süresi
    train_time = time.time() - start_time
    print(f"Eğitim Süresi: {train_time:.2f} saniye")
    
    # Test
    y_true = []
    y_pred = []
    
    for x, y_true_val in zip(X_test, y_test):
        y_pred_val = rnn.forward(x)
        prediction = int(y_pred_val.item() >= 0.5)
        
        y_true.append(y_true_val)
        y_pred.append(prediction)
    
    # Metrikleri hesapla
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    print(f"Test Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 Score: {f1:.2%}")
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'train_time': train_time,
        'epochs': len(train_losses)
    }

def evaluate_pytorch_rnn():
    print("\n===== PyTorch RNN Modeli Değerlendirmesi =====")
    start_time = time.time()
    
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
            
            # Her 50 epokta bir bilgi yazdırma
            if epoch % 50 == 0:
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
    
    # Eğitim süresi
    train_time = time.time() - start_time
    print(f"Eğitim Süresi: {train_time:.2f} saniye")
    
    # Test değerlendirmesi
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor)
        predictions = (y_pred >= 0.5).float()
        
        # Metrikleri hesaplama
        y_true = y_test_tensor.numpy().flatten()
        y_pred_np = predictions.numpy().flatten()
        
        accuracy = accuracy_score(y_true, y_pred_np)
        precision = precision_score(y_true, y_pred_np, zero_division=0)
        recall = recall_score(y_true, y_pred_np, zero_division=0)
        f1 = f1_score(y_true, y_pred_np, zero_division=0)
        
        print(f"Test Accuracy: {accuracy:.2%}")
        print(f"Precision: {precision:.2%}")
        print(f"Recall: {recall:.2%}")
        print(f"F1 Score: {f1:.2%}")
    
    return {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'train_time': train_time,
        'epochs': len(train_losses)
    }

def plot_comparison(custom_results, pytorch_results):
    # Loss karşılaştırması
    plt.figure(figsize=(12, 5))
    
    # Eğitim epoklarını normalize et (farklı uzunlukta olabilirler)
    custom_epochs = np.linspace(1, custom_results['epochs'], len(custom_results['train_losses']))
    pytorch_epochs = np.linspace(1, pytorch_results['epochs'], len(pytorch_results['train_losses']))
    
    plt.subplot(1, 2, 1)
    plt.plot(custom_epochs, custom_results['train_losses'], 'r-', label='Manuel RNN')
    plt.plot(pytorch_epochs, pytorch_results['train_losses'], 'b-', label='PyTorch RNN')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Eğitim Kaybı Karşılaştırması')
    plt.legend()
    plt.grid(True)
    
    # Accuracy karşılaştırması
    plt.subplot(1, 2, 2)
    plt.plot(custom_epochs, custom_results['train_accuracies'], 'r-', label='Manuel RNN')
    plt.plot(pytorch_epochs, pytorch_results['train_accuracies'], 'b-', label='PyTorch RNN')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Eğitim Doğruluğu Karşılaştırması')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('images/training_comparison.png')
    plt.show()
    
    # Test metrikleri karşılaştırması
    metrics = ['test_accuracy', 'precision', 'recall', 'f1']
    metric_names = ['Test Accuracy', 'Precision', 'Recall', 'F1 Score']
    
    custom_values = [custom_results[m] for m in metrics]
    pytorch_values = [pytorch_results[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, custom_values, width, label='Manuel RNN')
    plt.bar(x + width/2, pytorch_values, width, label='PyTorch RNN')
    
    plt.ylabel('Değer')
    plt.title('Model Performans Karşılaştırması')
    plt.xticks(x, metric_names)
    plt.legend()
    plt.grid(True, axis='y')
    
    # Değerleri çubukların üzerine yaz
    for i, v in enumerate(custom_values):
        plt.text(i - width/2, v + 0.01, f'{v:.2%}', ha='center')
    
    for i, v in enumerate(pytorch_values):
        plt.text(i + width/2, v + 0.01, f'{v:.2%}', ha='center')
    
    plt.tight_layout()
    plt.savefig('images/metrics_comparison.png')
    plt.show()
    
    # Eğitim süresi karşılaştırması
    plt.figure(figsize=(8, 5))
    plt.bar(['Manuel RNN', 'PyTorch RNN'], 
            [custom_results['train_time'], pytorch_results['train_time']])
    plt.ylabel('Süre (saniye)')
    plt.title('Eğitim Süresi Karşılaştırması')
    plt.grid(True, axis='y')
    
    # Değerleri çubukların üzerine yaz
    plt.text(0, custom_results['train_time'] + 0.5, f"{custom_results['train_time']:.2f}s", ha='center')
    plt.text(1, pytorch_results['train_time'] + 0.5, f"{pytorch_results['train_time']:.2f}s", ha='center')
    
    plt.tight_layout()
    plt.savefig('images/time_comparison.png')
    plt.show()

if __name__ == "__main__":
    custom_results = evaluate_custom_rnn()
    pytorch_results = evaluate_pytorch_rnn()
    
    plot_comparison(custom_results, pytorch_results)
    
    print("\n===== Model Karşılaştırması =====")
    print(f"Manuel RNN toplam epoch: {custom_results['epochs']}")
    print(f"PyTorch RNN toplam epoch: {pytorch_results['epochs']}")
    print(f"Manuel RNN eğitim süresi: {custom_results['train_time']:.2f} saniye")
    print(f"PyTorch RNN eğitim süresi: {pytorch_results['train_time']:.2f} saniye")
    print(f"Manuel RNN test doğruluğu: {custom_results['test_accuracy']:.2%}")
    print(f"PyTorch RNN test doğruluğu: {pytorch_results['test_accuracy']:.2%}")
    
    winner_accuracy = "Manuel RNN" if custom_results['test_accuracy'] > pytorch_results['test_accuracy'] else "PyTorch RNN"
    winner_time = "Manuel RNN" if custom_results['train_time'] < pytorch_results['train_time'] else "PyTorch RNN"
    
    print(f"\nEn yüksek test doğruluğuna sahip model: {winner_accuracy}")
    print(f"En hızlı eğitim süresine sahip model: {winner_time}")
