import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import copy
import numpy as np

class NetApproximator(nn.Module):
    def __init__(self, input_dim = 1, output_dim = 1, hidden_dim = 32):
        '''近似价值函数
        Args:
            input_dim: 输入层的特征数 int
            output_dim: 输出层的特征数 int
        '''
        super(NetApproximator, self).__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, output_dim)
    
    
    def _prepare_data(self, x, requires_grad = True):   #为什么requires_grad = False也能进行训练
        '''将numpy格式的数据转化为Torch的Variable
        '''
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if isinstance(x, int): # 同时接受单个数据
            x = torch.Tensor([[x]])
        x.requires_grad_ = requires_grad
        x = x.float()   # 从from_numpy()转换过来的数据是DoubleTensor形式
        if x.data.dim() == 1:
            x = x.unsqueeze(0)
        return x


    def forward(self, x):
        '''前向运算，根据网络输入得到网络输出
        '''
        x = self._prepare_data(x)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        y_pred = self.linear3(x)
        return y_pred

    

        
    def fit(self, loss, optimizer=None, 
                  epochs=1, learning_rate=1e-4):
        '''通过训练更新网络参数来拟合给定的输入x和输出y
        '''
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)
        if epochs < 1:
            epochs = 1


        for t in range(epochs):
            
            optimizer.zero_grad() # 梯度重置，准备接受新梯度值
            loss.backward() # 反向传播
            optimizer.step() # 更新权重
        
    
    
    def clone(self):
        '''返回当前模型的深度拷贝对象
        '''
        return copy.deepcopy(self)


if __name__ =="__main__":
    from microgrid import MicroGrid

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    critic_model_path = './ddqn_model.ckpt'

    policy_net = NetApproximator(27, 48, hidden_dim=128).to(device)
    env = MicroGrid()

    state = env.reset()
    print(state)

    def predict(state):
        with torch.no_grad():
            state = torch.tensor(
                np.array(state), device=device, dtype=torch.float32)
            q_values = policy_net(state)
            print(q_values)
            action = q_values.max(1)[1].item()
            # action = int(np.argmax(q_values.data.numpy()))
        return action
    print(predict(state))