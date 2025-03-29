import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import autoencoders
from Trainers import freeze_encoder
 
class Classifier(nn.Module):
    def __init__(self, input_size=128, hidden_layers=[64], output_size=10, dropout_rate=0.0):
        super(Classifier, self).__init__()

        layers = []
        prev_size = input_size

        # Input - an ENCODED img (N, 128) 
        for hidden_layer in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_layer))
            layers.append(nn.ReLU())
            #layers.append(nn.LeakyReLU())
            layers.append(nn.BatchNorm1d(hidden_layer))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_layer

        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)
   
    def forward(self, x):
        return self.model(x)

class clfTrainer(nn.Module):
    def __init__(self, classifier, encoder, dl_train, dl_test, hparams, freeze_encoder, device):
        super().__init__()
    
        self.classifier = classifier
        self.encoder = encoder
        self.dl_train = dl_train
        self.dl_test = dl_test
        self.loss_fn = hparams['loss_fn']
        self.lr = hparams['learning_rate']
        self.weight_decay = hparams['weight_decay']
        self.num_epochs = hparams['num_epochs']
        self.freeze_encoder = freeze_encoder
        self.device = device

        if self.freeze_encoder:
            self.optimizer = hparams['optimizer'](self.classifier.parameters(), lr=self.lr,
                                                  weight_decay=self.weight_decay)
        else:
            self.optimizer = hparams['optimizer'](list(self.encoder.parameters()) +
                                                   list(self.classifier.parameters()),
                                                  lr=self.lr, weight_decay=self.weight_decay)

    def trainClassifier(self):
       
        if self.freeze_encoder:
            freeze_encoder(self.encoder)
            
        train_acc = []
        test_acc = []
        
        for epoch in range(self.num_epochs):
            
            self.classifier.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for img, label in self.dl_train:
                img, label = img.to(self.device), label.to(self.device)
        
                if self.freeze_encoder:
                    with torch.no_grad():
                        self.encoder.eval()
                        encoded_img = self.encoder(img)  
                else:
                    self.encoder.train()
                    encoded_img = self.encoder(img)
                
                encoded_img = encoded_img.view(encoded_img.size(0), -1)
                output = self.classifier(encoded_img)
                loss = self.loss_fn(output, label)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
                running_loss += loss.item()

                _, predicted = torch.max(output, 1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

            train_loss = running_loss / len(self.dl_train)
            train_accuracy = 100 * correct / total
            train_acc.append(train_accuracy)
            print(f"Epoch {epoch + 1}:")
            print(f"    Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

            self.classifier.eval()
            self.encoder.eval()
            test_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for img, label in self.dl_test:
                    img, label = img.to(self.device), label.to(self.device)
                    encoded_img = self.encoder(img)
                    encoded_img = encoded_img.view(encoded_img.size(0), -1)
                    output = self.classifier(encoded_img)
                    loss = self.loss_fn(output, label)
                    test_loss += loss.item()

                    _, predicted = torch.max(output, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()

            test_loss = test_loss / len(self.dl_test)
            test_accuracy = 100 * correct / total
            test_acc.append(test_accuracy)
            print(f"    Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

        return train_acc, test_acc


    
    def evalClassifier(self, dl_val):

        self.classifier.eval()
        self.encoder.eval() 

        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad(): 
            for img, label in dl_val:
                img, label = img.to(self.device), label.to(self.device)
                encoded_img = self.encoder(img)
                encoded_img = encoded_img.view(encoded_img.size(0), -1)
                output = self.classifier(encoded_img)
                
                loss = self.loss_fn(output, label)
                val_loss += loss.item()

                # Compute accuracy
                _, predicted = torch.max(output, dim=1)
                total += label.size(0)
                correct += (predicted == label).sum().item()

        val_loss /= len(dl_val)
        val_accuracy = 100 * correct / total
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        return val_loss, val_accuracy
