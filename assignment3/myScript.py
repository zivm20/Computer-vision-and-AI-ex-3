import sys
from PIL import Image

import torch
#assert '.'.join(torch.__version__.split('.')[:2]) == '1.4'
import torch.nn as nn
import torch.nn.functional as F  # useful stateless functions
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np


#similar to the check_accuracy function in pytorch notebook
def check_accuracy(loader,device, model,verbose=0):
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        if verbose>1:
            print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

        return acc
    

#similar to the train function in pytorch notebook
def train(model,loader_train,loader_val,device, optimizer,lossFunction=F.cross_entropy, epochs=1, verbose = 0, print_every=100):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.
    
    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for
    
    Returns: train accuracy, validation accuracy
    """
    
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        if verbose>1:
            print('')
            print("epoch:",e)
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=torch.float32)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = lossFunction(scores, y)

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
                if verbose>1:
                    print('Iteration %d, loss = %.4f' % (t, loss.item()))
                    print('train:')
                    train_acc = check_accuracy(loader_train,device,model,verbose=verbose+1)
                    print('val:')
                    val_acc = check_accuracy(loader_val,device, model,verbose=verbose+1)
                    print('train accuracy: (%.2f), val accuracy: (%.2f)' % (100 * train_acc,100*val_acc))

    if verbose > 0:    
        print('train loss = %.4f' % (loss.item()))
    train_acc = check_accuracy(loader_train,device, model,verbose=verbose-1)
    val_acc = check_accuracy(loader_val,device, model,verbose=verbose-1)
    print('train accuracy: (%.2f), val accuracy: (%.2f)' % (100 * train_acc,100*val_acc))
    return train_acc,val_acc



def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)

def classify_image(model,device,image_path,transform):
    # Load and preprocess the input image
    image = Image.open(image_path).convert('RGB')
    
    image = transform(image).unsqueeze(0)

    # Forward pass through the model
    with torch.no_grad():
        x = image.to(device=device, dtype=torch.float32)
        output = model(x)
        

    # Get the predicted class label
    _, predicted_idx = torch.max(output, 1)
    predicted_label = class_labels[predicted_idx.item()]

    return predicted_label

if __name__ == '__main__':
    # Check if the image path is provided as a command-line argument
    mode = 'train'
    if len(sys.argv) == 2:
        mode = 'predict'
        
        # Get the image path from the command-line argument
        image_path = sys.argv[1]
    USE_GPU = True
    
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        
        
    # Define the transformation to apply to the input image
    transform = T.Compose([
                T.ToTensor(),
                T.Resize((16,16)),
                T.Grayscale()
            ])
    
    if mode == 'predict':
        # Load the pre-trained model
        model = torch.load('model.pt',map_location=device)
        model = model.to(device=device)
        class_labels = ['Arborio', 'Basmati', 'Ipsala','Jasmine','Karacadag']

        # Perform image classification
        predicted_label = classify_image(model,device,image_path,transform)
        print('Predicted label:', predicted_label)
    else:
        try:
            rice_data = dset.ImageFolder('./CV7062610/datasets/rice/Rice_Image_Dataset',transform=transform)
        except:
            print("cannot load from './CV7062610/datasets/rice/Rice_Image_Dataset'")
            exit(-1)
        rice_train, rice_val,rice_test = random_split(rice_data,[0.6,0.25,0.15],torch.Generator().manual_seed(13))
        loader_train = DataLoader(rice_train, batch_size=64)
        loader_val = DataLoader(rice_val, batch_size=64)
        loader_test = DataLoader(rice_test, batch_size=64)
        model_params = []
        

        model_params.append(nn.Conv2d(1,8,(5,5),padding = 5//2))
        model_params.append(nn.BatchNorm2d(8))
        model_params.append(nn.ReLU())

        model_params.append(nn.Conv2d(8,8,(5,5),padding = 5//2))
        model_params.append(nn.BatchNorm2d(8))
        model_params.append(nn.ReLU())

        model_params.append(nn.Conv2d(8,8,(5,5),padding = 5//2))
        model_params.append(nn.BatchNorm2d(8))
        model_params.append(nn.ReLU())

        model_params.append(nn.MaxPool2d(2))
        H = 1 + ((16-2)//2)
        W = 1 + ((16-2)//2)
        model_params.append(Flatten())

        model_params.append(nn.Linear(8*H*W,64))
        model_params.append(nn.BatchNorm1d(64))
        model_params.append(nn.ReLU())
        model_params.append(nn.Linear(64,5))

        model = nn.Sequential(*model_params)
        optimizer = optim.SGD(model.parameters(),lr=3e-2,momentum=0.9, nesterov=True)
        
        train_acc, val_acc = train(model,loader_train,loader_val,device, optimizer,epochs=2,verbose=2,print_every=500)
    
        print("done training")
        print('train accuracy: (%.2f), val accuracy: (%.2f)' % (100 * train_acc,100*val_acc))
        
        test_acc = check_accuracy(loader_test,device, model)
        print('test accuracy: (%.2f)' % (100*test_acc))

        torch.save(model, "model.pt")
        
