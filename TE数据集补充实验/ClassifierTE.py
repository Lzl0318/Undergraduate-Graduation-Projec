import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, recall_score, precision_score

'''################################    固定随机参数    #################################'''


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(47)

'''################################    导入数据和数据封装    #################################'''
train_data = np.load('./TE数据/train_data.npy')
test_data = np.load('./TE数据/test_data.npy')
train_label = np.load('./TE数据/train_label.npy').squeeze(1)
test_label = np.load('./TE数据/test_label.npy').squeeze(1)

train_data = torch.from_numpy(train_data).float()
train_label = torch.from_numpy(train_label).float().type(torch.LongTensor)
test_data = torch.from_numpy(test_data).float()
test_label = torch.from_numpy(test_label).float().type(torch.LongTensor)


train_dataset = TensorDataset(train_data, train_label)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

'''################################    分类模型    #################################'''


class Net(nn.Module):
    def __init__(self, n_input, n_output):
        super(Net, self).__init__()
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
    
    
def train():
    
    #实例化网络    
    model = Net(52, 22)
   
    #定义损失函数和超参数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss() 

    model.train()
    
    for i in range(300):
        for x, y in train_loader:
            out = model(x)
            loss = criterion(out, y)
            #反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        out = model(test_data)
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        
        target_y = test_label.data.numpy()
        accuracy = sum(pred_y == target_y)/(len(test_label))
        
        print('epoch = {}, 精度 = {}'.format(i, accuracy))

    # 模型最后测试
    model.eval()
    pre = model(test_data)
    prediction = torch.max(F.softmax(pre), 1)[1]
    pred_y = prediction.data.numpy().squeeze()
    target_y = test_label.data.numpy()
    print('准确率={:.4f}'.format(sum(pred_y == target_y)/4400))
    print('precision={:.4f}'.format(precision_score(target_y, pred_y, average='macro')))
    print('recall={:.4f}'.format(recall_score(target_y, pred_y, average='macro')))
    print('f1={:.4f}'.format(f1_score(target_y, pred_y, average='macro')))
    torch.save(model.state_dict(), './model.pt')


if __name__ == "__main__":
    train()

















