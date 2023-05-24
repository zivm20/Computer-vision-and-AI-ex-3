import sys
import torchvision.datasets as dset
import torchvision.transforms as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torch

# GLOBAL VARIABLES
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("running on: gpu")
else:
    device = torch.device('cpu')
    print("running on: cpu")
dtype = torch.float32
print_every = 500
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)
transform = T.Compose([
                T.ToTensor(),
                T.Grayscale()
            ])
rice_data = dset.ImageFolder('Rice_Image_Dataset',transform=transform)
rice_train, rice_val,rice_test = random_split(rice_data,[0.2,0.10,0.70],torch.Generator().manual_seed(69))
loader_train = DataLoader(rice_train, batch_size=64)
loader_val = DataLoader(rice_val, batch_size=64)
loader_test = DataLoader(rice_test, batch_size=64)

def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))

def train(model, optimizer, epochs=1):
    for e in range(epochs):
        print("epoc: " + str(1 + e))
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if t % print_every == 0:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy(loader_val, model)
                print()

def create_model():
    H = 1 + ((25-2)//2)
    W = 1 + ((25-2)//2)

    model = nn.Sequential(
        nn.Conv2d(1,8,(10,10), stride=10),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.Conv2d(8,8,(3,3),padding = 3//2),
        nn.BatchNorm2d(8),
        nn.ReLU(),
        nn.MaxPool2d(2),
        Flatten(),
        nn.Linear(8*H*W,5)
    )
    return model
    

if __name__ == '__main__':
    if len(sys.argv) == 2:
        model = torch.load('model.pt',map_location=device)
        model = model.to(device=device)
        class_labels = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

        # Perform image classification
        predicted_label = classify_image(model,device,image_path,transform)
        print('Predicted label:', predicted_label)
    else:
        model = create_model()
        optimizer = optim.SGD(model.parameters(),lr=2e-2,momentum=0.95, nesterov=True)

        # move the model parameters to CPU/GPU
        model = model.to(device=device)
        train(model, optimizer, epochs=3)
        check_accuracy(loader_test, model)
        
        torch.save(model, "model.pt")
        print("model saved to file")
