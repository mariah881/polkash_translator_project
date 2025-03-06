#Script contains the setup and training process
import torch
import torch.optim as optim
import torch.nn as nn
from model import create_model
from preprocessing import load_data
import json

def setup_and_train():
  """
  The function sets up and trains the Neural Machine Translation (NMT) seq2seq model.
  It initializes the model, optimizer, loss function,
  and then trains the model on the training data.

  It saves the best performing model based on the best result of validation loss.
  """
  #Defining GPU
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  train_loader, test_loader, INPUT_DIM, OUTPUT_DIM = load_data()

  # dim size comes from pre-training loading the data. we keep track of this and if it changes
  # we need to change it here as well.
  model = create_model(input=56241, output=60132, device=device)
 
  # Optimizer and loss
  optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
  criterion = nn.CrossEntropyLoss(ignore_index=0)

  #Training
  train_model(model, train_loader, test_loader, optimizer, criterion, device, n_epochs=5)

  return model


def train_model(model, train_loader, test_loader, optimizer, criterion, device, n_epochs=5, clip=1.0, patience=2):
    """
    The function trains the seq2seq NMT model using the data from previously defined corpus and optimization settings.
    Args:
        model: The PyTorch seq2seq NMT model to train.
        train_loader: DataLoader for the training dataset.
        test_loader: DataLoader for the validation dataset.
        optimizer: The optimizer to use (defined below).
        criterion: The loss function to use (defined below).
        device: The device to use for training ('cuda' or 'cpu').
        n_epochs: The number of training epochs. (adjusted to avoid excessive running time)
        clip: Gradient clipping threshold.
        patience: The number of epochs with no improvement after which training will be stopped.

    Returns:
        None. The function saves the best model checkpoint to 'best-model.pt'.

    """

    best_valid_loss = float('inf')
    early_stopping_counter = 0

    # Printing epoch count
    for epoch in range(n_epochs):
        print(f'Epoch count: {epoch+1}/{n_epochs}')

        model.train()
        epoch_loss = 0

        for i, (src, trg) in enumerate(train_loader):
            src, trg = src.to(device), trg.to(device)

            # Reseting gradient to zero to prevent accumulation of gradients between batches
            optimizer.zero_grad()

            # Model's forward pass
            output = model(src, trg)

            # Reshaping output and target
            output_dim = output.shape[-1]
            output = output[:, 1:].contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # Loss calculation
            loss = criterion(output, trg)

            # Backward pass
            loss.backward()

            # Gradient clipping to limit the size of gradient updates
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            # Updating parameters
            optimizer.step()

            epoch_loss += loss.item()

            # Printing loss for every 100 batches for possible hyperparameter adjusting
            if i % 100 == 0:
                print(f'Batch {i}, Loss: {loss.item():.4f}')

        # Average train loss calculation (not per batch)
        train_loss = epoch_loss / len(train_loader)
        print(f'Train loss: {train_loss:.4f}')

        # Model evaluation using test set
        model.eval()
        epoch_loss = 0

        # No gradient calculation for validation
        with torch.no_grad():
            for src, trg in test_loader:
                src, trg = src.to(device), trg.to(device)

                # Model's forward pass
                output = model(src, trg)

                # Reshaping output and target
                output_dim = output.shape[-1]
                output = output[:, 1:].contiguous().view(-1, output_dim)
                trg = trg[:, 1:].contiguous().view(-1)

                # Loss calculation
                loss = criterion(output, trg)
                epoch_loss += loss.item()

        # Average validation loss calculation
        valid_loss = epoch_loss / len(test_loader)
        print(f'Validation loss: {valid_loss:.4f}')

        # Early stopping logic
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            early_stopping_counter = 0
            # Saving the best model
            torch.save(model.state_dict(), 'best-model.pt')
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping triggered.")
                break

        torch.save(model.state_dict(), 'best-model.pt')

setup_and_train()

#Saving metadata
import json
metadata = {
    "learning_rate": 0.0005,
    "dropout": 0.3,
    "batch_size": 16,
    "epochs": 5,
    "optimizer": "Adam",
    "loss_function": "CrossEntropyLoss"
}

with open("training_metadata.json", "w") as f:
    json.dump(metadata, f)



