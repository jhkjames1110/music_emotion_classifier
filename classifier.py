import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import librosa
import librosa.display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import M2EClassifier

# Using librosa to convert .wav files to data we can feed into an ANN model.

# function to link each of the wave files to their "target" emotion
# returning a pandas dataframe
def create_dataframe(dir):
    target = []
    audio = []
    for i in os.listdir(dir):
        for j in os.listdir(dir + "/" + str(i)):
            for k in os.listdir(dir + "/" + i + "/" + j):
                target.append(str(j))
                audio.append(dir + "/" + str(i) + "/" + j + "/" + k)
        df = pd.DataFrame(columns=["audio", "target"])
        df["audio"] = audio
        df["target"] = target
    return df

# function for Music Information Retrieval
def extract_features(data, sample_rate):
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data, hop_length=20).T, axis=0)
    result = np.hstack((result, zcr))

    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate, n_fft=20, hop_length=20).T, axis=0)
    result = np.hstack((result, chroma_stft))

    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=20).T, axis=0)
    result = np.hstack((result, mfcc))

    rms = np.mean(librosa.feature.rms(y=data, frame_length=100).T, axis=0)
    result = np.hstack((result, rms))

    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate, hop_length=20).T, axis=0)
    result = np.hstack((result, mel))

    return result

def feature_extractor(path):
    data, sample_rate = librosa.load(path)
    res1 = extract_features(data, sample_rate)
    result = np.array(res1)
    return result


# Training function
def train(model, criterion, optimizer, train_loader, val_loader, num_epochs=250):
    model.train()
    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # L2 regularization
            # l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
            # loss = loss + l2_lambda * l2_norm
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            _, labels_max = torch.max(labels, 1)
            correct_train += (predicted == labels_max).sum().item()
            total_train += labels.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct_train / total_train
        train_loss_history.append(epoch_loss)
        train_acc_history.append(epoch_acc)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                
                # l2_norm = sum(p.pow(2.0).sum() for p in model.parameters())
                # loss = loss + l2_lambda * l2_norm
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                _, labels_max = torch.max(labels, 1)
                correct_val += (predicted == labels_max).sum().item()
                total_val += labels.size(0)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct_val / total_val
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        
        model.train()
        
        # Log the training and validation metrics to TensorBoard
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Loss/validation', val_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        writer.add_scalar('Accuracy/validation', val_acc, epoch)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Acc: {val_acc:.4f}")
    
    return train_loss_history, val_loss_history, train_acc_history, val_acc_history

# Evaluate the model on the test set
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            _, labels_max = torch.max(labels, 1)
            correct += (predicted == labels_max).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return accuracy


if __name__=="__main__":
    # To load the saved dataframe from the output directory:
    dataframe = pd.read_csv("../data/features.csv")
    
    # determining the features and the target values.
    X = np.array(dataframe.drop(columns=["target"]))
    y = np.array(dataframe["target"])
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    encoder = OneHotEncoder()
    y_train = encoder.fit_transform(y_train.reshape(-1, 1)).toarray()
    y_test = encoder.fit_transform(y_test.reshape(-1, 1)).toarray()
    
    # Convert the data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    # Create DataLoader objects
    batch_size = 16
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
    # Instantiate the model, define the optimizer and the loss function
    model = M2EClassifier()
    
    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    
    # l2_lambda = 0.01 # regularization param
    
    # Initialize TensorBoard writer
    writer = SummaryWriter('runs/my_experiment')

    # Train the model
    num_epochs = 200
    train_loss, val_loss, train_acc, val_acc = train(model, criterion, optimizer, train_loader, test_loader, num_epochs)
    torch.save(model.state_dict(), 'm2e_classifier.pth') # save model
    
    # Evaluate the model
    test_accuracy = evaluate(model, test_loader)
    print(f'Test Accuracy: {test_accuracy:.4f}')
    
    # Plot loss and accuracy
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper left')
    plt.savefig('train_loss.png')
    
    plt.clf()

    plt.plot(train_acc, label='train_accuracy')
    plt.plot(val_acc, label='val_accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper left')
    plt.savefig('train_accuracy.png')
        
    writer.close()