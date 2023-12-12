import torch
import random


class Model():
    def __init__(self):
        self.Actor = torch.nn.Sequential(
            torch.nn.Linear(4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2),
            torch.nn.Softmax(dim=1),
        )

        self.Critic = torch.nn.Sequential(
            torch.nn.Linear(4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 1),
        )
        self.Actor(torch.randn(2, 4))
        self.Critic(torch.randn(2, 4))

    def get_action(self, state):
        state = torch.FloatTensor(state).reshape(1, 4)
        prob = self.Actor(state)
        action = random.choices(range(2), weights=prob[0].tolist(), k=1)[0]
        return action

    # def get_action(self, states):
    #     states = torch.FloatTensor(states)  # 不再需要 reshape
    #     probs = self.Actor(states)
    #     actions = [random.choices(range(2), weights=prob.tolist(), k=1)[0] for prob in probs]
    #     return actions

model = Model()
# action = model.get_action([[1, 1, 1, 1], [1, 1, 1, 1]])
for i in range(100):
    action = model.get_action([1, 1, 1, 1])
    print("action is ", action)