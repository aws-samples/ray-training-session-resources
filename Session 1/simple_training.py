# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

# Defining model structure
model = nn.Sequential(
    nn.Linear(764, 100),
    nn.ReLU(),
    nn.Linear(100,50),
    nn.ReLU(),
    nn.Linear(50,10),
    nn.Sigmoid()
).to("cpu")

# Define training hyperparameters
learning_rate = 0.001
num_epochs = 10
batch_size = 32

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define custom dataset
class CustomDataset(Dataset):
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.LongTensor(targets)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Define training loop
def train(model, train_loader, criterion, optimizer, num_epochs: int) -> None:
    model.train() # Set model to training mode

    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            # Set parameter gradients to zero
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 100 == 9: # print every 100 mini-batches
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

        print(f'Finished Epoch {epoch+1}')

def main():
    # Load data
    print("Loading sample data...")
    df = pd.read_csv("sample_data.csv")
    print("Dataset read successfully")

    # Prepare features and target
    X = df.drop('target', axis=1).values
    y = df['target'].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)

    # Create datasets
    train_dataset = CustomDataset(X_train, y_train)
    test_dataset = CustomDataset(X_test, y_test)

    # Set training parameters
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 10
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # Not considering evaluation dataset here for simplicity

    # Train the model
    print("Starting Training...")
    start_train = time.time()
    train(model, train_loader, criterion, optimizer, num_epochs)
    
    # Save model
    torch.save(model.state_dict(), "./model.pth")  # or .pt extension
    print("Model saved as model.pth")
    print(f"Training completed in {time.time() - start_train} seconds")


# Training execution
if __name__ == "__main__":
    start = time.time()
    main()
    print(f"Total run time: {time.time() - start} seconds")