import torch
import torch.nn as nn
import torch.optim as optim

class GeneratorPretrainer:
    def __init__(self, generator, loss_fn, optimizer):
        self.generator = generator
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def pretrain(self, train_dataset, num_epochs):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator.to(device)
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            # Iterate over the pretraining dataset
            for attributes, target_features in train_dataset:
                attributes = attributes.to(device)
                target_features = target_features.to(device)
                
                # Zero the gradients
                self.optimizer.zero_grad()
                
                # Generate features
                generated_features = self.generator(torch.randn(attributes.size(0), latent_size).to(device), attributes)
                
                # Calculate the loss between the generated features and the target features
                loss = self.loss_fn(generated_features, target_features)
                
                # Backpropagation
                loss.backward()
                
                # Update generator weights
                self.optimizer.step()
                
                # Track the loss value for this batch
                epoch_loss += loss.item()
            
            # Print the average loss for this epoch
            avg_loss = epoch_loss / len(train_dataset)
            print("Epoch {}: Loss = {}".format(epoch+1, avg_loss))