"""
Part V. CIFAR-10 open-ended challenge
In this section, you can experiment with whatever ConvNet architecture you'd like on CIFAR-10.

Now it's your job to experiment with architectures, hyperparameters, loss functions, and optimizers to train a model that achieves at least 70% accuracy on the CIFAR-10 validation set within 10 epochs. You can use the check_accuracy and train functions from above. You can use either nn.Module or nn.Sequential API.

Describe what you did at the end of this notebook.

Here are the official API documentation for each component. One note: what we call in the class "spatial batch norm" is called "BatchNorm2D" in PyTorch.

Layers in torch.nn package: http://pytorch.org/docs/stable/nn.html
Activations: http://pytorch.org/docs/stable/nn.html#non-linear-activations
Loss functions: http://pytorch.org/docs/stable/nn.html#loss-functions
Optimizers: http://pytorch.org/docs/stable/optim.html
Things you might try:
Filter size: Above we used 5x5; would smaller filters be more efficient?
Number of filters: Above we used 32 filters. Do more or fewer do better?
Pooling vs Strided Convolution: Do you use max pooling or just stride convolutions?
Batch normalization: Try adding spatial batch normalization after convolution layers and vanilla batch normalization after affine layers. Do your networks train faster?
Network architecture: The network above has two layers of trainable parameters. Can you do better with a deep network? Good architectures to try include:
conv-relu-poolxN -> affinexM -> softmax or SVM
conv-relu-conv-relu-poolxN -> affinexM -> softmax or SVM
batchnorm-relu-convxN -> affinexM -> softmax or SVM
Global Average Pooling: Instead of flattening and then having multiple affine layers, perform convolutions until your image gets small (7x7 or so) and then perform an average pooling operation to get to a 1x1 image picture (1, 1 , Filter#), which is then reshaped into a (Filter#) vector. This is used in Google's Inception Network (See Table 1 for their architecture).
Regularization: Add l2 weight regularization, or perhaps use Dropout.
Tips for training
For each network architecture that you try, you should tune the learning rate and other hyperparameters. When doing this there are a couple important things to keep in mind:

If the parameters are working well, you should see improvement within a few hundred iterations
Remember the coarse-to-fine approach for hyperparameter tuning: start by testing a large range of hyperparameters for just a few training iterations to find the combinations of parameters that are working at all.
Once you have found some sets of parameters that seem to work, search more finely around these parameters. You may need to train for more epochs.
You should use the validation set for hyperparameter search, and save your test set for evaluating your architecture on the best parameters as selected by the validation set.
Going above and beyond
If you are feeling adventurous there are many other features you can implement to try and improve your performance. You are not required to implement any of these, but don't miss the fun if you have time!

Alternative optimizers: you can try Adam, Adagrad, RMSprop, etc.
Alternative activation functions such as leaky ReLU, parametric ReLU, ELU, or MaxOut.
Model ensembles
Data augmentation
New Architectures
ResNets where the input from the previous layer is added to the output.
DenseNets where inputs into previous layers are concatenated together.
This blog has an in-depth overview
Have fun and happy training!
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torch.nn.functional as F

import torchvision.datasets as dset
import torchvision.transforms as T
import torchvision as vision

import numpy as np
import matplotlib.pyplot as plt

NUM_TRAIN = 49000

# The torchvision.transforms package provides tools for preprocessing data
# and for performing data augmentation; here we set up a transform to
# preprocess the data by subtracting the mean RGB value and dividing by the
# standard deviation of each RGB value; we've hardcoded the mean and std.
transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])

# We set up a Dataset object for each split (train / val / test); Datasets load
# training examples one at a time, so we wrap each Dataset in a DataLoader which
# iterates through the Dataset and forms minibatches. We divide the CIFAR-10
# training set into train and val sets by passing a Sampler object to the
# DataLoader telling how it should sample from the underlying Dataset.
cifar10_train = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
                            transform=transform)
loader_train = DataLoader(cifar10_train, batch_size=64, 
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

cifar10_val = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
                        transform=transform)
loader_val = DataLoader(cifar10_val, batch_size=64, 
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))

cifar10_test = dset.CIFAR10('./cs231n/datasets', train=False, download=True, 
                            transform=transform)
loader_test = DataLoader(cifar10_test, batch_size=64)


device = torch.device('cuda') #!
print_every = 100
dtype = torch.float32

def save_checkpoint(checkpoint_path, model, optimizer):
    # state_dict: a Python dictionary object that:
    # - for a model, maps each layer to its parameter tensor;
    # - for an optimizer, contains info about the optimizer’s states and hyperparameters used.
    state = {
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)
    
def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

def train_part34(model, optimizer, epochs=1):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    losses = []
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                val_loss = check_accuracy_part34(loader_val, model)
                print()
                losses.append((loss, val_loss))
    return losses

def check_accuracy_part34(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')   
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            loss = F.cross_entropy(scores, y)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
        return loss


image_size = 32
#* (in + 2* padding - filter) / stride  +  1 
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),        
            nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(32 * 16 * 16, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            nn.Linear(1000, 10),
        )


    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        x = self.fc(x)
        return F.log_softmax(x,dim=1)

if True:
    model = vision.models.resnet18(pretrained=True)
    print(model)
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    print(num_features)
    model.fc = nn.Linear(num_features, 10)

    input_size = 224

    data_transforms = {
        'train': T.Compose([
            T.RandomResizedCrop(input_size),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': T.Compose([
            T.Resize(input_size),
            T.CenterCrop(input_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    cifar10_train = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
                                transform=data_transforms["train"])
    loader_train = DataLoader(cifar10_train, batch_size=64, 
                            sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))

    cifar10_val = dset.CIFAR10('./cs231n/datasets', train=True, download=True,
                            transform=data_transforms["val"])
    loader_val = DataLoader(cifar10_val, batch_size=64, 
                            sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))


    print(model)

    model.to(device)
    print("To Update:")
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad is True:
            params_to_update.append(param)
            print(name)

    optim = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
    train_part34(model, optim, epochs=2)
    input("End Part Model Load")

if False:
    model = MyModel()
    optim = optim.Adam(model.parameters(), lr=0.001)

    check_accuracy_part34(loader_val, model.to(device))

    checkpoint = torch.load("fin.pth")
    print(model.state_dict().keys())
    states_to_load = {}
    for name, param in checkpoint["state_dict"].items():
        if name.startswith("conv"):
            states_to_load[name] = param
    for c in states_to_load:
        print(c)
    print("Number of parameter variables to load:", len(states_to_load))
    model_state = model.state_dict()
    print("Number of parameter variables in the model:", len(model_state))

    model_state.update(states_to_load)

    model.load_state_dict(model_state)
    optim.load_state_dict(checkpoint["optimizer"])

    train_part34(model, optim)
    #check_accuracy_part34(loader_val, model)


    input("End Part load")
########################################
model = MyModel()
optim = optim.Adam(model.parameters(), lr=0.001)

for (x, y) in loader_train:
    #print(x.shape)
    model(x)
    break

losses = train_part34(model, optim, epochs=2)
save_checkpoint("fin.pth", model, optim)

train, val = list(zip(*losses))
tl, = plt.plot(train, 'b-', lw=3, label="train")
vl, = plt.plot(val, 'o-', lw=3, label="validation")
plt.legend(handles=[tl,vl])
plt.savefig("a.png")