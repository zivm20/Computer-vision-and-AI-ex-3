import sys
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

def flatten(x):
    N = x.shape[0] # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image
class Flatten(nn.Module):
    def forward(self, x):
        return flatten(x)

def classify_image(image_path):
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
    if len(sys.argv) != 2:
        print('Usage: python script.py <image_path>')
        sys.exit(1)
    USE_GPU = True
    # Get the image path from the command-line argument
    image_path = sys.argv[1]
    if USE_GPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Define the transformation to apply to the input image
    
    transform = transforms.Compose([
                transforms.Resize((64,64)),
                transforms.ToTensor()
                ])

    # Load the pre-trained model
    model = torch.load('model.pt')
    model = model.to(device=device)
    


    class_labels = ['Arborio', 'Basmati', 'Ipsala','Jasmine','Karacadag']

    # Perform image classification
    predicted_label = classify_image(image_path)
    print('Predicted label:', predicted_label)