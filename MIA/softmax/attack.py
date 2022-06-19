import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s' % device)

# set argparse
parser = argparse.ArgumentParser(description='Training for Model Invertion Attack.')
parser.add_argument('--dataset', type=str, default="MNIST",
                    help='choose MNIST or AT&T dataset.')
args = parser.parse_args()

# define dimensions
if args.dataset == 'MNIST':
    input_dim = 28 * 28
    output_dim = 10
    root_path = '../data/mnist'
elif args.dataset == 'AT&T':
    input_dim = 112 * 92
    output_dim = 40
    root_path = '../data/face'


# net define
class Softmax(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Softmax, self).__init__()
        self.regression = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        output = self.regression(x)

        return output


def mi_face(label_index, model, num_iterations, gradient_step):
    model.to(device)
    model.eval()

    # initialize two 112 * 92 tensors with zeros
    if args.dataset == 'MNIST':
        tensor = torch.zeros(28, 28).unsqueeze(0).to(device)
        image = torch.zeros(28, 28).unsqueeze(0).to(device)
    elif args.dataset == 'AT&T':
        tensor = torch.zeros(112, 92).unsqueeze(0).to(device)
        image = torch.zeros(112, 92).unsqueeze(0).to(device)


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
            img_show = reconstruction.squeeze().detach().numpy()
            # for ii in range(img_show.shape[0]):
            #     for jj in range(img_show.shape[1]):
            #         if img_show[ii][jj] < 0:
            #             img_show[ii][jj] = 0
            #         else:
            #             img_show[ii][jj] = 255

            # add both images to the plot
            axs[i, j].imshow(original, cmap='gray')
            axs[i + 1, j].imshow(img_show, cmap='gray')
            axs[i, j].axis('off')
            axs[i + 1, j].axis('off')

    # plot reconstructed image
    fig.suptitle('Images reconstructed with ' + str(
        iterations) + ' iterations of mi_face. Find the reconstruction below each row with train set samples.',
                         fontsize=20)
    fig.savefig('../results/softmax_result_'+args.dataset+'.png', dpi=100)
    plt.show()
    print('\nReconstruction Results can be found in results folder')


model = Softmax(input_dim, output_dim)
model.load_state_dict(torch.load('../pretrain_models/softmax_model_'+args.dataset+'.pt'))
perform_attack_and_print_all_results(model, 100)
