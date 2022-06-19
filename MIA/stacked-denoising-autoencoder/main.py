import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import torch.optim as optim
import os
import random


# set argparse
parser = argparse.ArgumentParser(description='Training for Model Invertion Attack.')
parser.add_argument('--dataset', type=str, default="AT&T",
                    help='choose MNIST or AT&T dataset.')
parser.add_argument('--protection', type=int, default=1,
                    help='whether protect or not, 0 is not.')
args = parser.parse_args()

if args.dataset == 'MNIST':
    input_dim1 = 28
    input_dim2 = 28
    output_dim = 10
    root_path = '../data/mnist'
elif args.dataset == 'AT&T':
    input_dim1 = 112
    input_dim2 = 92
    output_dim = 40
    root_path = '../data/face'
    
    
class Layer1(nn.Module):
    def __init__(self, hidden_size):
        super(Layer1, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(input_dim1 * input_dim2, hidden_size),  # 112*92->1000
            nn.Sigmoid(),
            nn.Dropout(0.1)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_dim1 * input_dim2),
            nn.Sigmoid(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        batch_size = x.shape[0]  # [batch, 1, 112, 92]
        x = x.view(batch_size, -1)  # [batch, 112*92]

        encoded = self.encoder(x)  # [batch, 1000]
        decoded = self.decoder(encoded).reshape(batch_size, 1, input_dim1, input_dim2)  # [batch, 1, 112, 92]
        return encoded, decoded


class Layer2(nn.Module):
    def __init__(self, layer1, hidden_size):
        super(Layer2, self).__init__()
        self.layer1 = layer1
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(layer1.hidden_size, hidden_size),  # 1000->300
            nn.Sigmoid(),
            nn.Dropout(0.1)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, self.layer1.hidden_size),
            nn.Linear(self.layer1.hidden_size, input_dim1 * input_dim2),  # 300->1000->112*92
            nn.Sigmoid(),
            nn.Dropout(0.1)
        )

    def forward(self, x):
        # 保证前一层参数不变
        self.layer1.eval()
        batch_size = x.shape[0]  # [batch, 1, 112, 92]
        x = x.view(batch_size, -1)  # [batch, 112*92]
        x, _ = self.layer1(x)
        encoded = self.encoder(x)  # [batch, 300]
        decoded = self.decoder(encoded).reshape(batch_size, 1, input_dim1, input_dim2)  # [batch, 1, 112, 92]
        return encoded, decoded


class Layer3(nn.Module):
    def __init__(self, layer2, hidden_size):
        super(Layer3, self).__init__()
        self.layer2 = layer2
        self.hidden_size = hidden_size
        self.encoder = nn.Sequential(
            nn.Linear(layer2.hidden_size, hidden_size)  # 300->40
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, self.layer2.hidden_size),  # 40->300
            nn.Linear(self.layer2.hidden_size, self.layer2.layer1.hidden_size),  # 300->1000
            nn.Linear(self.layer2.layer1.hidden_size, input_dim1 * input_dim2),  # 1000->112*92
        )

    def forward(self, x):
        # 保证前一层参数不变
        self.layer2.eval()
        batch_size = x.shape[0]  # [batch, 1, 112, 92]
        x = x.view(batch_size, -1)  # [batch, 112*92]
        x, _ = self.layer2(x)
        encoded = self.encoder(x)  # [batch, 40]
        decoded = self.decoder(encoded).reshape(batch_size, 1, input_dim1, input_dim2)  # [batch, 1, 112, 92]
        return encoded, decoded


def get_one_hot(tensor, class_num, device):
    batch_size = tensor.shape[0]
    tensor = tensor.unsqueeze(dim=1)
    return torch.zeros(batch_size, class_num).to(device).scatter_(1, tensor, 1)


def train_layer(layer, k, device, trainloader):
    loss_fn = torch.nn.MSELoss().to(device)
    optimizer = optim.Adam(layer.parameters())
    for epoch in range(30):
        for img, label in trainloader:
            img, label = img.to(device), label.to(device)
            encoded, decoded = layer(img)
            optimizer.zero_grad()
            loss = loss_fn(decoded, img)
            loss.backward()
            optimizer.step()
        # save img
        fake_img = decoded.cpu().data
        real_img = img.cpu().data
        save_image(fake_img, './'+args.dataset+'_IMG/train-epochs-{}-fake_img.jpg'.format(epoch), nrow=8)
        save_image(real_img, './'+args.dataset+'_IMG/train-epochs-{}-real_img.jpg'.format(epoch), nrow=8)
        print("epoch:%d,loss:%f" % (epoch, loss))
    torch.save(layer.state_dict(), './sdae_layers/'+args.dataset+'_layer%d.pkl' % k)


def train_classifier(model, device, trainloader):
    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters())
    for epoch in range(30):
        for img, label in trainloader:
            img, label = img.to(device), label.to(device)
            encoded, decoded = model(img)
            optimizer.zero_grad()
            loss = loss_fn(encoded, label)
            loss.backward()
            optimizer.step()
        print("epoch:%d,loss:%f" % (epoch, loss))
    torch.save(model.state_dict(), '../pretrain_models/sdae_model_'+args.dataset+'.pt')


def eval_layer(layer, device, validloader):
    layer.eval()
    loss_fn = torch.nn.MSELoss().to(device)
    for i, (img, label) in enumerate(validloader):
        img, label = img.to(device), label.to(device)
        encoded, decoded = layer(img)

        loss = loss_fn(decoded, img)

        # save img
        fake_img = decoded.cpu().data
        real_img = img.cpu().data
        save_image(fake_img, './'+args.dataset+'_IMG/valid-epochs-{}-fake_img.jpg'.format(i), nrow=8)
        save_image(real_img, './'+args.dataset+'_IMG/valid-epochs-{}-real_img.jpg'.format(i), nrow=8)
        print("valid:%d,loss:%f" % (i, loss))


def eval_classifier(model, device, validloader):
    model.eval()
    for i, (img, label) in enumerate(validloader):
        img, label = img.to(device), label.to(device)
        encoded, decoded = model(img)
        pred = torch.argmax(encoded, dim=1)
        print('pred:{}|real:{}'.format(pred.cpu().numpy(), label.cpu().numpy()))


def mi_face(label_index, model, num_iterations, gradient_step, device):
    model.to(device)
    model.eval()

    # initialize two 112 * 92 tensors with zeros
    tensor = torch.zeros(input_dim1, input_dim2).unsqueeze(0).to(device)
    image = torch.zeros(input_dim1, input_dim2).unsqueeze(0).to(device)

    # initialize with infinity
    min_loss = float("inf")

    for i in range(num_iterations):
        tensor.requires_grad = True

        # get the prediction probs
        pred, _ = model(tensor)

        # calculate the loss and gardient for the class we want to reconstruct
        crit = nn.CrossEntropyLoss()
        loss = crit(pred, torch.tensor([label_index]).to(device))

        print('Loss: ' + str(loss.item()))

        loss.backward()

        with torch.no_grad():
            # apply gradient descent
            tensor = (tensor - gradient_step * tensor.grad)

        # set image = tensor only if the new loss is the min from all iterations
        if loss < min_loss:
            min_loss = loss
            image = tensor.detach().clone().to('cpu')

    return image


def perform_attack_and_print_all_results(model, iterations, device):

    gradient_step_size = 0.1

    # print all pictures
    # create figure
    if args.dataset == 'MNIST':
        fig, axs = plt.subplots(2, 10)
        fig.set_size_inches(20, 12)
    elif args.dataset == 'AT&T':
        fig, axs = plt.subplots(8, 10)
        fig.set_size_inches(20, 24)

    random.seed(7)
    count = 0
    for i in range(0, int(output_dim/5), 2):
        for j in range(10):
            # get random validation set image from respective class
            count += 1
            print('\nReconstructing Class ' + str(count))

            ran = random.randint(1, 2)
            path = root_path + '/s0' + str(count) + '/' + str(
                ran) + '.pgm' if count < 10 else root_path + '/s' + str(count) + '/' + str(ran) + '.pgm'

            with open(path, 'rb') as f:
                original = plt.imread(f)

            # reconstruct respective class
            reconstruction = mi_face(count - 1, model, iterations, gradient_step_size, device)
            img_show = reconstruction.squeeze().detach().numpy()
            for ii in range(img_show.shape[0]):
                for jj in range(img_show.shape[1]):
                    if img_show[ii][jj] < 0:
                        img_show[ii][jj] = 0
                    else:
                        img_show[ii][jj] = 255

            # add both images to the plot
            axs[i, j].imshow(original, cmap='gray')
            axs[i + 1, j].imshow(img_show, cmap='gray')
            axs[i, j].axis('off')
            axs[i + 1, j].axis('off')

    # plot reconstructed image
    fig.suptitle('Images reconstructed with ' + str(
        iterations) + ' iterations of mi_face. Find the reconstruction below each row with train set samples.',
                         fontsize=20)
    fig.savefig('../results/sdae_result_'+args.dataset+'.png', dpi=100)
    plt.show()
    print('\nReconstruction Results can be found in results folder')


def main():
    # transfrom, wee need grayscale to convert the images to 1 channel
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    # load dataset
    dataset = datasets.ImageFolder(root_path, transform=transform)

    # split dataset: 3 images of every class as validation set
    i = [i for i in range(len(dataset)) if i % 10 > 3]
    i_val = [i for i in range(len(dataset)) if i % 10 <= 3]

    # load data
    BATCH_SIZE = 4
    train_dataset = torch.utils.data.Subset(dataset, i)
    train_data_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)

    validation_dataset = torch.utils.data.Subset(dataset, i_val)
    validation_data_loader = data.DataLoader(validation_dataset, shuffle=True, batch_size=BATCH_SIZE)

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device: %s' % device)

    # train three layers
    layer1 = Layer1(1000).to(device)
    if os.path.exists('./sdae_layers/'+args.dataset+'_layer1.pkl'):
        layer1.load_state_dict(torch.load('./sdae_layers/'+args.dataset+'_layer1.pkl'))
    else:
        print('\nlayer1 begin training\n')
        train_layer(layer1, 1, device, train_data_loader)

    layer2 = Layer2(layer1, 300).to(device)
    if os.path.exists('./sdae_layers/' + args.dataset + '_layer2.pkl'):
        layer1.load_state_dict(torch.load('./sdae_layers/' + args.dataset + '_layer2.pkl'))
    else:
        print('\nlayer2 begin training\n')
        train_layer(layer2, 2, device, train_data_loader)

    layer3 = Layer3(layer2, 40).to(device)
    if os.path.exists('./sdae_layers/' + args.dataset + '_layer3.pkl'):
        layer1.load_state_dict(torch.load('./sdae_layers/' + args.dataset + '_layer3.pkl'))
    else:
        print('\nlayer3 begin training\n')
        train_layer(layer3, 3, device, train_data_loader)

    # eval layer
    print('\nlayer3 begin evaluating\n')
    eval_layer(layer3, device, validation_data_loader)

    # train classify
    print('\nclassifier begin training\n')
    train_classifier(layer3, device, train_data_loader)

    # eval classify
    print('\nclassifier begin evaluating')
    eval_classifier(layer3, device, validation_data_loader)

    # attack sdae
    print('\nattack begin\n')
    perform_attack_and_print_all_results(layer3, 100, device)


if __name__ == '__main__':
    main()


