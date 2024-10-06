import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

class Classic_CNN(nn.Module):
    def __init__(self,input_channels=1,output_channels=10):
        super(Classic_CNN,self).__init__()
        # input x*16*16
        self.conv1 = nn.Conv2d(in_channels=input_channels,out_channels=16,kernel_size=3,stride=1,padding=1)
        # input 16*16*16
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1)

        self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.fc1 = None
        self.fc2 = nn.Linear(128,512)
        self.fc3 = nn.Linear(512,output_channels)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        if self.fc1 is None:
            # Dynamically define fc1 based on input size
            self.fc1 = nn.Linear(x.size(1) * x.size(2) * x.size(3), 128).to(x.device)
        x = x.view(-1,x.size(1) * x.size(2) * x.size(3))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def fit(self, train_loader,val_loader, criterion, optimizer, epochs = 10,device='cpu'):
        self.to(device)
        train_losses = []
        val_losses = []
        val_accuracies = []
        for epoch in range(epochs):
            self.train()
            running_loss = 0.
            for images,labels in train_loader:
                images,labels = images.to(device),labels.to(device)

                # clear the gradients
                optimizer.zero_grad()

                # forward pass
                outputs = self(images)

                # compute loss
                loss = criterion(outputs,labels)

                # back propagation
                loss.backward()
                # update weights and biases
                optimizer.step()

                running_loss+=loss.item()
            train_losses.append(running_loss / len(train_loader))
            val_loss, val_accuracy = self.validate(val_loader, criterion, device=device, verbose=False)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader):.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        
        self.plot_metrics(train_losses, val_losses, val_accuracies)

    def validate(self, test_loader, criterion, device='cpu', verbose=True):
        self.eval()
        correct = 0
        total = 0
        running_loss = 0.0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
        accuracy = 100 * correct / total
        avg_loss = running_loss / len(test_loader)
        if verbose:
            print(f"Test Accuracy: {accuracy:.2f}%")
            print(f"Validation Loss: {avg_loss:.4f}")
            print("Classification Report:")
            print(classification_report(all_labels, all_predictions))
        return avg_loss, accuracy

    def predict(self, image, device='cpu'):
        self.eval()
        image = image.to(device)
        with torch.no_grad():
            outputs = self(image)
            _, predicted = torch.max(outputs.data, 1)
        return predicted

    def plot_metrics(self, train_losses, val_losses, val_accuracies):
        # Plotting loss and accuracy curves
        plt.figure(figsize=(18, 5))
        plt.subplot(1, 3, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Validation Loss Curve')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.plot(val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Validation Accuracy Curve')
        plt.legend()
        
        plt.show()

