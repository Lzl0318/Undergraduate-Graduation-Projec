import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, f1_score, recall_score
from scipy import spatial
from sklearn.preprocessing import MinMaxScaler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s' % device)

# set argparse
parser = argparse.ArgumentParser(description='Training for Model Invertion Attack.')
parser.add_argument('--protection', type=int, default=0,
                    help='whether protect or not, 0 is not.')
args = parser.parse_args()


# define net
class MLP(nn.Module):
    def __init__(self, n_input, n_output):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(n_input, 512)
        self.hidden2 = nn.Linear(512, 256)
        self.hidden3 = nn.Linear(256, 128)
        self.predict = nn.Linear(128, n_output)

    def forward(self, x):
        out = self.hidden1(x)
        out = torch.relu(out)
        out = self.hidden2(out)
        out = torch.relu(out)
        out = self.hidden3(out)
        out = torch.relu(out)
        out = self.predict(out)

        return out


def mi_face(label_index, model, num_iterations, gradient_step):
    model.to(device)
    model.eval()

    # initialize two 52 tensors with zeros
    tensor = 0.5*torch.ones(52).unsqueeze(0).to(device)
    image = 0.5*torch.ones(52).unsqueeze(0).to(device)

    # tensor = torch.zeros(52).unsqueeze(0).to(device)
    # image = torch.zeros(52).unsqueeze(0).to(device)

    # initialize with infinity
    min_loss = float("inf")

    for i in range(num_iterations):
        tensor.requires_grad = True

        # get the prediction probs
        pred = model(tensor)

        # calculate the loss and gardient for the class we want to reconstruct
        crit = nn.CrossEntropyLoss()
        loss = crit(pred, torch.tensor([label_index]).to(device))

        # print('Loss: ' + str(loss.item()))

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
    random.seed(7)

    # initialize minmax scaler
    scaler = MinMaxScaler()

    # average evaluation
    avg_mae = 0
    avg_mse = 0
    avg_cos = 0

    for j in range(22):
        print('\nReconstructing Class ' + str(j+1))
        train_data = np.load('./TE数据/train_data.npy')
        original = train_data[10+600*j]

        # reconstruct respective class
        reconstruction = mi_face(j, model, iterations, gradient_step_size)
        reconstruction = reconstruction.squeeze().detach().numpy().reshape(-1, 1)
        reconstruction = scaler.fit_transform(reconstruction)
        reconstruction = reconstruction.squeeze(1)

        # evaluation two vector
        mae = mean_absolute_error(original, reconstruction)
        mse = mean_squared_error(original, reconstruction)
        cos = 1 - spatial.distance.cosine(original, reconstruction)

        print('\nThe mean_absolute_error of reconstruction vector and original vector is {}'.format(mae))
        print('\nThe mean_squared_error of reconstruction vector and original vector is {}'.format(mse))
        print('\nThe cosine_similarity of reconstruction vector and original vector is {}'.format(cos))

        avg_mae = (avg_mae + mae)
        avg_mse = (avg_mse + mse)
        avg_cos = (avg_cos + cos)

        # print(original)
        # print(reconstruction)
    print('\nThe average mean_absolute_error of reconstruction vector and original vector is {}'.format(avg_mae/22))
    print('\nThe average mean_squared_error of reconstruction vector and original vector is {}'.format(avg_mse/22))
    print('\nThe average cosine_similarity of reconstruction vector and original vector is {}'.format(avg_cos/22))


model = MLP(52, 22).to(device)
parameters = torch.load('./model.pt')
# -------------------------加噪声----------------------------
if args.protection == 1:
    weight1 = parameters['hidden1.weight'].cpu().numpy()
    weight1_new = weight1 + np.random.normal(weight1.mean(), weight1.std() * 1, weight1.shape)
    parameters['hidden1.weight'] = torch.tensor(weight1_new).float().to(device)

    bias1 = parameters['hidden1.bias'].cpu().numpy()
    bias1_new = bias1 + np.random.normal(bias1.mean(), bias1.std() * 1, bias1.shape)
    parameters['hidden1.bias'] = torch.tensor(bias1_new).float().to(device)

    weight2 = parameters['hidden2.weight'].cpu().numpy()
    weight2_new = weight2 + np.random.normal(weight2.mean(), weight2.std() * 1, weight2.shape)
    parameters['hidden2.weight'] = torch.tensor(weight2_new).float().to(device)

    bias2 = parameters['hidden2.bias'].cpu().numpy()
    bias2_new = bias2 + np.random.normal(bias2.mean(), bias2.std() * 1, bias2.shape)
    parameters['hidden2.bias'] = torch.tensor(bias2_new).float().to(device)

    weight1 = parameters['hidden1.weight'].cpu().numpy()
    weight1_new = weight1 + np.random.normal(weight1.mean(), weight1.std() * 1, weight1.shape)
    parameters['hidden1.weight'] = torch.tensor(weight1_new).float().to(device)

    bias1 = parameters['hidden1.bias'].cpu().numpy()
    bias1_new = bias1 + np.random.normal(bias1.mean(), bias1.std() * 1, bias1.shape)
    parameters['hidden1.bias'] = torch.tensor(bias1_new).float().to(device)

    weight4 = parameters['predict.weight'].cpu().numpy()
    weight4_new = weight4 + np.random.normal(weight4.mean(), weight4.std() * 1, weight4.shape)
    parameters['predict.weight'] = torch.tensor(weight4_new).float().to(device)

    bias4 = parameters['predict.bias'].cpu().numpy()
    bias4_new = bias4 + np.random.normal(bias4.mean(), bias4.std() * 1, bias4.shape)
    parameters['predict.bias'] = torch.tensor(bias4_new).float().to(device)

# -----------------------------------------------------
model.load_state_dict(parameters)
perform_attack_and_print_all_results(model, 100)


# ---------------测试加噪声之后的模型的准确率---------------
# test_data = np.load('./TE数据/test_data.npy')
# test_label = np.load('./TE数据/test_label.npy').squeeze(1)
# test_data = torch.from_numpy(test_data).float()
# test_label = torch.from_numpy(test_label).float().type(torch.LongTensor)
#
# model.eval()
# model.cpu()
# pre = model(test_data)
# prediction = torch.max(F.softmax(pre), 1)[1]
# pred_y = prediction.data.numpy().squeeze()
# target_y = test_label.data.numpy()
# print('准确率={:.4f}'.format(sum(pred_y == target_y)/4400))
# print('precision={:.4f}'.format(precision_score(target_y, pred_y, average='macro')))
# print('recall={:.4f}'.format(recall_score(target_y, pred_y, average='macro')))
# print('f1={:.4f}'.format(f1_score(target_y, pred_y, average='macro')))


