import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import spatial


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


def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)


# def label_to_onehot(target, num_classes=22):
#     target = torch.unsqueeze(target, 1)
#     onehot_target = torch.zeros(target.size(0), num_classes, device=target.device)
#     onehot_target.scatter_(1, target, 1)
#     return onehot_target
#
#
# def cross_entropy_for_onehot(pred, target):
#     return torch.mean(torch.sum(- target * F.log_softmax(pred, dim=-1), 1))


# 加载模型
model = MLP(52, 22)
torch.manual_seed(1234)
model.apply(weights_init)

# 加载数据
data = np.load('./TE数据/train_data.npy')
label = np.load('./TE数据/train_label.npy')
data = torch.from_numpy(data[100]).float().unsqueeze(0)
label = torch.from_numpy(label[100]).float().type(torch.LongTensor)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 计算原始梯度
pred = model(data)
y = criterion(pred, label)
dy_dx = torch.autograd.grad(y, model.parameters())
original_dy_dx = list((_.detach().clone() for _ in dy_dx))

# generate dummy data and label
dummy_data = torch.from_numpy(np.random.normal(0.5, 0.5, data.size())).float().requires_grad_(True)
dummy_label = torch.Tensor([11]).requires_grad_(True)

# print first dummy data
print(dummy_data[0])

# 优化器
optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

# begin loop
history = []
for iters in range(100):
    def closure():
        optimizer.zero_grad()

        dummy_pred = model(dummy_data)
        dummy_loss = criterion(dummy_pred, dummy_label.long())
        dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)

        grad_diff = 0
        for gx, gy in zip(dummy_dy_dx, original_dy_dx):
            grad_diff += ((gx - gy) ** 2).sum()
        grad_diff.backward()

        return grad_diff


    optimizer.step(closure)
    if iters % 10 == 0:
        current_loss = closure()
        print(iters, "%.4f" % current_loss.item())
        history.append(dummy_data[0])

# evaluation
for vec in history:
    mae = mean_absolute_error(data.numpy(), vec.numpy())
    mse = mean_squared_error(data.numpy(), vec.numpy())
    cos = 1 - spatial.distance.cosine(data.numpy(), vec.numpy())

    print('\nThe mean_absolute_error of reconstruction vector and original vector is {}'.format(mae))
    print('\nThe mean_squared_error of reconstruction vector and original vector is {}'.format(mse))
    print('\nThe cosine_similarity of reconstruction vector and original vector is {}'.format(cos))