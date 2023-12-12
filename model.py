import torch
import random
from env import MyWrapper

#定义模型
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

    def save(self, actor_path, critic_path):
        torch.save(self.Actor.state_dict(), actor_path)
        torch.save(self.Critic.state_dict(), critic_path)

    def load(self, actor_path, critic_path):
        self.Actor.load_state_dict(torch.load(actor_path))
        self.Critic.load_state_dict(torch.load(critic_path))


    # get_action([1, 2, 3, 4])
    def get_action(self, state):
        # state = torch.FloatTensor(state).reshape(1, 4)
        # prob = self.Actor(state)
        # action = random.choices(range(2), weights=prob[0].tolist(), k=1)[0]
        # return action
        states = torch.FloatTensor(state).reshape(len(state), 4)
        probs = self.Actor(states)
        actions = [random.choices(range(2), weights=prob.tolist(), k=1)[0] for prob in probs]
        return actions

    def get_data(self, env):
        self.env = env
        states = []
        rewards = []
        actions = []
        next_states = []
        overs = []

        # 初始化游戏
        state, info = self.env.reset()

        # 玩到游戏结束为止
        over = [False, False, False]
        while not any(over):
            # 根据当前状态得到一个动作
            action_n = self.get_action(state)

            # 执行动作,得到反馈
            next_state, reward, over, _, _ = self.env.step(action_n)

            # 记录数据样本
            states.append(state[0])
            states.append(state[1])
            states.append(state[2])
            rewards.append(reward[0])
            rewards.append(reward[1])
            rewards.append(reward[2])

            actions.append(action_n[0])
            actions.append(action_n[1])
            actions.append(action_n[2])
            next_states.append(next_state[0])
            next_states.append(next_state[1])
            next_states.append(next_state[2])
            overs.append(over[0])
            overs.append(over[1])
            overs.append(over[2])

            # 更新游戏状态,开始下一个动作
            state = next_state

        # [b, 4]
        states = torch.FloatTensor(states).reshape(-1, 4)
        # [b, 1]
        rewards = torch.FloatTensor(rewards).reshape(-1, 1)
        # [b, 1]
        actions = torch.LongTensor(actions).reshape(-1, 1)
        # [b, 4]
        next_states = torch.FloatTensor(next_states).reshape(-1, 4)
        # [b, 1]
        overs = torch.LongTensor(overs).reshape(-1, 1)

        return states, rewards, actions, next_states, overs




