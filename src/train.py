import torch
from torch.utils.data.dataset import random_split


from data_preprocessing import PreprocessedDataLoader
from model import ClassifierModel

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Data preprocessing
dataloader = PreprocessedDataLoader("data")

# Train Eval split
train_size = int(0.9 * len(dataloader))
eval_size = len(dataloader)-train_size

train_data, eval_data = random_split(dataloader, [train_size, eval_size])

# Model

model = ClassifierModel()

# Loss and optimizer

loss_criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Train model
epochs = 15
size = len(dataloader.dataset)

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}\n-------------------------------")
    model.train()

    for batch, inputs, labels in enumerate(dataloader.dataset):
        inputs, labels = inputs.to(device), labels.to(device)

        # Prediction loss
        outputs = model(inputs)
        loss = loss_criterion(outputs, labels)

        #Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 6 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


