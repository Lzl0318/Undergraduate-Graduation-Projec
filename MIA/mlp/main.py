import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F

import torchvision.transforms as transforms
import torchvision.datasets as datasets

# set seeds
SEED = 12
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# set argparse
parser = argparse.ArgumentParser(description='Training for Model Invertion Attack.')
parser.add_argument('--dataset', type=str, default="MNIST",
                    help='choose MNIST or AT&T dataset.')
args = parser.parse_args()


# net define
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 3000)
        self.fc2 = nn.Linear(3000, output_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        x = F.sigmoid(self.fc1(x))
        output = self.fc2(x)

        return output


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for (x, y) in iterator:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        y_pred = model(x)
        loss = criterion(y_pred, y)
        acc = calculate_accuracy(y_pred, y)

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device):
    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            loss = criterion(y_pred, y)
            acc = calculate_accuracy(y_pred, y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def train_target_model(epochs):
    # transfrom, wee need grayscale to convert the images to 1 channel
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    # define dimensions
    if args.dataset == 'MNIST':
        input_dim = 28 * 28
        output_dim = 10
        root_path = '../data/mnist'
    elif args.dataset == 'AT&T':
        input_dim = 112 * 92
        output_dim = 40
        root_path = '../data/face'

    # load dataset
    dataset = datasets.ImageFolder(root_path, transform=transform)

    # split dataset: 3 images of every class as validation set
    i = [i for i in range(len(dataset)) if i % 10 > 3]
    i_val = [i for i in range(len(dataset)) if i % 10 <= 3]

    # load data
    BATCH_SIZE = 16
    train_dataset = torch.utils.data.Subset(dataset, i)
    train_data_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)

    validation_dataset = torch.utils.data.Subset(dataset, i_val)
    validation_data_loader = data.DataLoader(validation_dataset, batch_size=BATCH_SIZE)

    # create model
    mynet = MLP(input_dim, output_dim)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: %s' % device)
    mynet = mynet.to(device)

    # set criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mynet.parameters(), lr=0.001)

    # main loop
    best_valid_loss = float('inf')

    print('---Target Model Training Started---')
    for epoch in range(epochs):

        train_loss, train_acc = train(mynet, train_data_loader, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(mynet, validation_data_loader, criterion, device)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

        print(f'Epoch: {epoch + 1:02}')
        print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')

    torch.save(mynet.state_dict(), '../pretrain_models/mlp_model_'+args.dataset+'.pt')
    print('---Target Model Training Done---')


train_target_model(20)
