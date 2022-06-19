import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s' % device)

# set argparse
parser = argparse.ArgumentParser(description='Training for Model Invertion Attack.')
parser.add_argument('--dataset', type=str, default="AT&T",
                    help='choose MNIST or AT&T dataset.')
parser.add_argument('--protection', type=int, default=1,
                    help='whether protect or not, 0 is not.')
args = parser.parse_args()

# define dimensions
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


def mi_face(label_index, model, num_iterations, gradient_step):
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
        pred = model(tensor)

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


def perform_attack_and_print_all_results(model, iterations):

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
    # 记录重建图像与原图像之间的均方差
    mse_loss = 0
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
            reconstruction = mi_face(count - 1, model, iterations, gradient_step_size)
            reconstruction = reconstruction.squeeze().detach().numpy()
            # for ii in range(reconstruction.shape[0]):
            #     for jj in range(reconstruction.shape[1]):
            #         if reconstruction[ii][jj] < 0:
            #             reconstruction[ii][jj] = 0
            #         else:
            #             reconstruction[ii][jj] = 255

            # scaler = MinMaxScaler()
            # original = scaler.fit_transform(original)
            # reconstruction = scaler.fit_transform(reconstruction)


            # add both images to the plot
            axs[i, j].imshow(original, cmap='gray')
            axs[i + 1, j].imshow(reconstruction, cmap='gray')
            axs[i, j].axis('off')
            axs[i + 1, j].axis('off')

            mse_loss = mse_loss + mean_squared_error(original, reconstruction)

    # plot reconstructed image
    fig.suptitle('Images reconstructed with ' + str(
        iterations) + ' iterations of mi_face. Find the reconstruction below each row with train set samples.',
                         fontsize=20)
    fig.savefig('../results/mlp_result_protect_'+str(args.protection)+'_'+args.dataset+'.png', dpi=100)
    plt.show()
    print('\nReconstruction Results can be found in results folder')
    print('\nThe MSE of reconstruction face and original face is {}'.format(mse_loss))
    print(original)
    print(reconstruction)


model = MLP(input_dim1*input_dim2, output_dim).to(device)
parameters = torch.load('../pretrain_models/mlp_model_'+args.dataset+'.pt')
# -----------------------------------------------------
if args.protection == 1:
    weight1 = parameters['fc1.weight'].cpu().numpy()
    weight1_new = weight1 + np.random.normal(weight1.mean(), weight1.std() * 3, weight1.shape)
    parameters['fc1.weight'] = torch.tensor(weight1_new).float().to(device)
    
    bias1 = parameters['fc1.bias'].cpu().numpy()
    bias1_new = bias1 + np.random.normal(bias1.mean(), bias1.std() * 3, bias1.shape)
    parameters['fc1.bias'] = torch.tensor(bias1_new).float().to(device)
    
    weight2 = parameters['fc2.weight'].cpu().numpy()
    weight2_new = weight2 + np.random.normal(weight2.mean(), weight2.std() * 3, weight2.shape)
    parameters['fc2.weight'] = torch.tensor(weight2_new).float().to(device)

    bias2 = parameters['fc2.bias'].cpu().numpy()
    bias2_new = bias2 + np.random.normal(bias2.mean(), bias2.std() * 3, bias2.shape)
    parameters['fc2.bias'] = torch.tensor(bias2_new).float().to(device)
    
# -----------------------------------------------------
model.load_state_dict(parameters)
perform_attack_and_print_all_results(model, 100)
# ---------------测试加噪声之后的模型的准确率---------------


def calculate_accuracy(y_pred, y):
    top_pred = y_pred.argmax(1, keepdim=True)
    correct = top_pred.eq(y.view_as(top_pred)).sum()
    acc = correct.float() / y.shape[0]
    return acc


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


transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
dataset = datasets.ImageFolder(root_path, transform=transform)
i = [i for i in range(len(dataset)) if i % 10 > 3]
i_val = [i for i in range(len(dataset)) if i % 10 <= 3]
BATCH_SIZE = 16
validation_dataset = torch.utils.data.Subset(dataset, i_val)
validation_data_loader = data.DataLoader(validation_dataset, batch_size=BATCH_SIZE)
for epoch in range(20):
    valid_loss, valid_acc = evaluate(model, validation_data_loader, nn.CrossEntropyLoss(), device)

    print(f'Epoch: {epoch + 1:02}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc * 100:.2f}%')
