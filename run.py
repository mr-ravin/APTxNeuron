import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm 
import pandas as pd
import os

parser = argparse.ArgumentParser(prog='MNIST digit classification using a fully-connected feedforward neural network based on APTx Neuron architecture.')
parser.add_argument("--mode", "-m", default="train")
parser.add_argument("--total_epoch", "-tep", type=int, default=20)
parser.add_argument("--lr", "-lr", type=float, default=4e-3)
parser.add_argument("--load_model_weights_path", "-pth", default="./weights/aptx_neural_network_11.pt") # 11th Epoch showed 96.69 % accuracy on test set.
parser.add_argument("--device", "-d", default="cpu") # we used cpu for experimentation on MNIST dataset

args = parser.parse_args()
MODE = args.mode
TOTAL_EPOCH = args.total_epoch
DEVICE = args.device
INFERENCE_MODEL_WEIGHTS = args.load_model_weights_path
LR = args.lr
CSV_STORE_PATH = "./result/"

# -----------------------------------
# APTx Neuron (Single Unit)
# -----------------------------------
class APTx_Neuron(nn.Module):
    def __init__(self, input_dim, is_alpha_trainable=True):
        super(APTx_Neuron, self).__init__()
        if is_alpha_trainable:
            self.alpha = nn.Parameter(torch.randn(input_dim)) 
        else:
            self.register_buffer('alpha', torch.ones(input_dim)) # To reduce trainable parameters from 3n + 1 to 2n + 1 (where n is the input dimension), replace with: self.alpha = torch.ones(input_dim)  # (fix α_i = 1 to make it non-trainable)
        self.beta  = nn.Parameter(torch.randn(input_dim))
        self.gamma = nn.Parameter(torch.randn(input_dim))
        self.delta = nn.Parameter(torch.zeros(1))

    def forward(self, x):  # x: [batch_size, input_dim]
        nonlinear = (self.alpha + torch.tanh(self.beta * x)) * self.gamma * x
        y = nonlinear.sum(dim=1, keepdim=True) + self.delta
        return y

# -----------------------------------
# APTx Layer (Multiple Neurons)
# -----------------------------------
class APTxLayer(nn.Module):
    def __init__(self, input_dim, output_dim, is_alpha_trainable=True):
        super(APTxLayer, self).__init__()
        self.neurons = nn.ModuleList([APTx_Neuron(input_dim, is_alpha_trainable) for _ in range(output_dim)])

    def forward(self, x):  # x: [batch_size, input_dim]
        outputs = [neuron(x) for neuron in self.neurons]  # list of [batch_size, 1]
        return torch.cat(outputs, dim=1)  # [batch_size, output_dim]

# -----------------------------------
# Full APTxNet Model
# -----------------------------------
class APTxNet(nn.Module):
    def __init__(self, input_dim=784, hidden1=128, hidden2=64, hidden3=32, num_classes=10):
        super(APTxNet, self).__init__()
        self.aptx1 = APTxLayer(input_dim, hidden1)
        self.aptx2 = APTxLayer(hidden1, hidden2)
        self.aptx3 = APTxLayer(hidden2, hidden3)
        self.fc_out = nn.Linear(hidden3, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)     # Flatten: [batch_size, 784]
        x = self.aptx1(x)             # [batch_size, 128]
        x = self.aptx2(x)             # [batch_size, 64]
        x = self.aptx3(x)             # [batch_size, 32]
        logits = self.fc_out(x)       # [batch_size, 10]
        return logits                 # raw scores (logits)

# -----------------------------------
# Training Function
# -----------------------------------
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    correct = 0
    total_loss = 0
    loss = 0
    with tqdm(train_loader, unit=" Train batch") as tepoch:
        tepoch.set_description(f"Train Epoch:")
        for data, target in tepoch:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)  # raw logits
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            # Apply softmax here for probabilities
            probs = F.softmax(output.detach().cpu(), dim=1)         
            pred = probs.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    print(f"Epoch {epoch}: Train Loss = {total_loss / len(train_loader):.4f}")        
    # Save recent model weights here
    torch.save(model.state_dict(), "./weights/aptx_neural_network_"+str(epoch)+".pt")
    print(">>> Saved model weights in file: ./weights/aptx_neural_network_"+str(epoch)+".pt")
    accuracy = 100. * correct / len(train_loader.dataset)
    print(f"Train Accuracy: {accuracy:.2f}%")
    return round(loss.item(),4), round(accuracy,4)

# -----------------------------------
# Testing Function (with Softmax)
# -----------------------------------
def test(model, device, test_loader, criterion):
    model.eval()
    correct = 0
    loss = 0
    with torch.no_grad():
        with tqdm(test_loader, unit=" Test batch") as tepoch:
            tepoch.set_description(f"Test Epoch:")
            for data, target in tepoch:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                # Apply softmax here for probabilities
                probs = F.softmax(output, dim=1)         
                pred = probs.argmax(dim=1)
                correct += pred.eq(target).sum().item()
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f"Test Accuracy: {accuracy:.2f}%")
    return round(loss.item(),4), round(accuracy,4)

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Trainable Parameters: {total:,}")

# -----------------------------------
# Model Training Script
# -----------------------------------
def train_and_evaluate(device):
    # MNIST Data Loaders
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = DataLoader(datasets.MNIST(root='.', train=True, download=True, transform=train_transform),
                              batch_size=64, shuffle=True)
    test_loader = DataLoader(datasets.MNIST(root='.', train=False, transform=test_transform),
                             batch_size=1000, shuffle=False)

    # Model, Optimizer, Loss
    model = APTxNet().to(device)
    count_parameters(model)  # <<------ Print total parameters here
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.25)
    criterion = nn.CrossEntropyLoss()
    write_dict = {'epoch':[],'train_loss':[],'test_loss':[], 'train_accuracy':[], 'test_accuracy':[]}
    # Training Loop
    for epoch in range(1, TOTAL_EPOCH + 1):
        print("Epoch: ", epoch)
        write_dict['epoch'].append(epoch)
        train_loss, train_accuracy = train(model, device, train_loader, optimizer, criterion, epoch)
        lr_scheduler.step()
        test_loss, test_accuracy = test(model, device, test_loader, criterion)
        write_dict['train_loss'].append(train_loss)
        write_dict['test_loss'].append(test_loss)
        write_dict['train_accuracy'].append(train_accuracy)
        write_dict['test_accuracy'].append(test_accuracy)
    df = pd.DataFrame(write_dict)
    # Write the DataFrame to a CSV file
    csv_output_path = CSV_STORE_PATH+"train_mode_output.csv"
    print("Loss and Accuracy values are saved in: ", csv_output_path)
    df.to_csv(csv_output_path, index=False)


# -----------------------------------
# Model Inference Script
# -----------------------------------
def inference(model_weight_path, device):
    # MNIST Data Loaders
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    criterion = nn.CrossEntropyLoss()
    test_loader = DataLoader(datasets.MNIST(root='.', train=False, transform=test_transform),
                             batch_size=1000, shuffle=False)

    # Model, Optimizer, Loss
    model = APTxNet().to(device)
    model.load_state_dict(torch.load(model_weight_path, map_location=device, weights_only=True))
    count_parameters(model)  # <<------ Print total parameters here
    write_dict = {'epoch':['inference'], 'test_loss':[], 'test_accuracy':[]}
    # Training Loop

    test_loss, test_accuracy = test(model, device, test_loader, criterion)
    write_dict['test_loss'].append(test_loss)
    write_dict['test_accuracy'].append(test_accuracy)
    df = pd.DataFrame(write_dict)
    # Write the DataFrame to a CSV file
    csv_output_path = CSV_STORE_PATH+"inference_mode_output.csv"
    print("Loss and Accuracy values are saved in: ", csv_output_path)
    df.to_csv(csv_output_path, index=False)
    print("Test accuracy: ", test_accuracy)

if __name__ == "__main__":
    print("Removing previously stored weights: ./weights/*pt")
    if MODE.lower() == "train":
        os.system("rm ./weights/*pt")
        train_and_evaluate(DEVICE)
    elif MODE.lower() in ["inference", "infer", "test", "eval", "evaluate"]:
        inference(INFERENCE_MODEL_WEIGHTS, DEVICE)
