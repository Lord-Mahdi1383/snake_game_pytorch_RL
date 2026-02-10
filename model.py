import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class DQN(nn.model):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()
        self.lin1 = nn.Linear(input_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.lin2(2)
        return x


    def save():
        pass


class DQN_trainer():
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    
    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )


        pred = self.model(state)

        target = pred.clone()
        for idx in range(len(done)):
            q_new = reward[idx]
            if not done[idx]:
                q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = q_new 

        
        # loss and backpropagation
        self.optimizer.zero_grad()
        loss = self.criterion(target. pred)
        loss.backward()
        self.optimizer.step()
            