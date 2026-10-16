import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# DEVICE CONFIGURATION
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
input_size = 784 # 28 x 28 grid 
hidden_size = 500 # define the hidden layer
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.001

# MNIST dataset -- THIS IS THE IN-BUILT DATA SET FROM TORCHVISION, USE THIS WHEN TRAINING MODELS IF APPLICABLE

train_dataset = torchvision.datasets.MNIST(root='./data', 
                                            train=True,
                                            transform=transforms.ToTensor(), # SPECIFY A BUILT-IN TRANSFORM --> transform to tensor
                                            download=True)

test_dataset = torchvision.datasets.MNIST(root = './data',
                                            train=False,
                                            transform=transforms.ToTensor(),
                                            download=True)

# Data Loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, # for the train-loader, shuffle=True, the train-loader is just the wrapper for feeding data
                                            batch_size=batch_size, # define a batch size
                                            shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, # for the train-loader, shuffle=True
                                            batch_size=batch_size, # define a batch size
                                            shuffle=False)

examples = iter(test_loader)
example_data, example_target = next(examples)

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(example_data[i][0], cmap='gray')
plt.show()

# START MAKING THE NEURAL NET

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU() # this is one of the many activation functions --> ReLU, Sigmoid... 
        self.l2 = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation or softmax
        return out 

model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# THEN WRITE THE LOSS AND OPTIMISER --> use nn.CrossEntropyLoss() and there is a list of optimizers that can be used
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the Model 
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # origin shape: [100, 1, 28, 28]
        # resized: [100, 784]
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        # Forward pass and loss calculations
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimise --> optimizer.zero_grad() comes before loss backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item(): .4f}')


# You can then test the model without needing to compute the gradients:
with torch.no_grad():
    n_correct = 0
    n_samples = len(test_loader.dataset)

    for images, labels in test_loader:
        images = images.reshape(-1, 28 * 28).to(device)
        labels = labels.to(device)

        outputs = model(images)

        # max returns (output_value, index)
        _, predicted = torch.max(outputs, 1)
        n_correct += (predicted == labels).sum().item()

    acc = n_correct / n_samples # the accuracy of the model can be classified as a ratio between these two things.
    print(f'Accuracy of the network on the {n_samples} test images is {acc}')




