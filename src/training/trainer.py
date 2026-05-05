import torch
from tqdm import tqdm
from torch import nn, optim

# function trains a model using train and validation loaders, then saves weights
def train_model(model, train_loader, val_loader, device, epochs, save_path):

    # move model to device
    model = model.to(device)

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

    # loop over epochs
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        # train phase
        model.train()
        total_loss = 0

        # iterate over training batches with progress bar
        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)

            # reset gradients
            optimizer.zero_grad()

            # forward pass
            out = model(x)

            # compute loss
            loss = criterion(out, y)

            # backpropagation
            loss.backward()

            # update weights
            optimizer.step()

            # accumulate loss
            total_loss += loss.item()

        print("Train Loss:", total_loss / len(train_loader))

        # validation phase
        model.eval()
        correct, total = 0, 0

        # disable gradient computation
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)

                # get predictions
                preds = model(x).argmax(1)

                # count correct predictions
                correct += (preds == y).sum().item()
                total += y.size(0)

        print("Val Acc:", 100 * correct / total)

    # save trained model weights
    torch.save(model.state_dict(), save_path)
    print("Saved:", save_path)
