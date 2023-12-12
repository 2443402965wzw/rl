import gym
import torch
import random

#定义环境
class MyWrapper(gym.Wrapper):

    def __init__(self):
        env = gym.make('CartPole-v1')
        super().__init__(env)
        self.env = env
        self.step_n = 0

    def reset(self):
        state = self.env.reset()
        self.step_n = 0
        return state

    def step(self, action):
        state, reward, terminated, info = self.env.step(action)
        done = terminated
        self.step_n += 1
        if self.step_n >= 200:
            done = True
        return state, reward, done, info


env = MyWrapper()


#定义模型
model = torch.nn.Sequential(
    torch.nn.Linear(4, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 2),
    torch.nn.Softmax(dim=1),
)

model_td = sequential = torch.nn.Sequential(
    torch.nn.Linear(4, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 1),
)

model(torch.randn(2, 4)), model_td(torch.randn(2, 4))


#得到一个动作
def get_action(state):
    state = torch.FloatTensor(state).reshape(1, 4)
    #[1, 4] -> [1, 2]
    prob = model(state)

    #根据概率选择一个动作
    action = random.choices(range(2), weights=prob[0].tolist(), k=1)[0]

    return action

def get_data():
    states = []
    rewards = []
    actions = []
    next_states = []
    overs = []

    #初始化游戏
    state = env.reset()

    #玩到游戏结束为止
    over = False
    while not over:
        #根据当前状态得到一个动作
        action = get_action(state)

        #执行动作,得到反馈
        next_state, reward, over, _ = env.step(action)

        #记录数据样本
        states.append(state)
        rewards.append(reward)
        actions.append(action)
        next_states.append(next_state)
        overs.append(over)

        #更新游戏状态,开始下一个动作
        state = next_state

    #[b, 4]
    states = torch.FloatTensor(states).reshape(-1, 4)
    #[b, 1]
    rewards = torch.FloatTensor(rewards).reshape(-1, 1)
    #[b, 1]
    actions = torch.LongTensor(actions).reshape(-1, 1)
    #[b, 4]
    next_states = torch.FloatTensor(next_states).reshape(-1, 4)
    #[b, 1]
    overs = torch.LongTensor(overs).reshape(-1, 1)

    return states, rewards, actions, next_states, overs

def test(play):
    #初始化游戏
    state = env.reset()

    #记录反馈值的和,这个值越大越好
    reward_sum = 0

    #玩到游戏结束为止
    over = False
    while not over:
        #根据当前状态得到一个动作
        action = get_action(state)

        #执行动作,得到反馈
        state, reward, over, _ = env.step(action)
        reward_sum += reward

        # #打印动画
        # if play and random.random() < 0.2:  #跳帧
        #     display.clear_output(wait=True)
        #     show()

    return reward_sum

def train():
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer_td = torch.optim.Adam(model_td.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()

    #玩N局游戏,每局游戏训练一次
    for i in range(1000):
        #玩一局游戏,得到数据
        #states -> [b, 4]
        #rewards -> [b, 1]
        #actions -> [b, 1]
        #next_states -> [b, 4]
        #overs -> [b, 1]
        states, rewards, actions, next_states, overs = get_data()

        #计算values和targets
        #[b, 4] -> [b ,1]
        values = model_td(states)

        #[b, 4] -> [b ,1]
        targets = model_td(next_states) * 0.98
        #[b ,1] * [b ,1] -> [b ,1]
        targets *= (1 - overs)
        #[b ,1] + [b ,1] -> [b ,1]
        targets += rewards

        #时序差分误差
        #[b ,1] - [b ,1] -> [b ,1]
        delta = (targets - values).detach()

        #重新计算对应动作的概率
        #[b, 4] -> [b ,2]
        probs = model(states)
        #[b ,2] -> [b ,1]
        probs = probs.gather(dim=1, index=actions)

        #根据策略梯度算法的导函数实现
        #只是把公式中的reward_sum替换为了时序差分的误差
        #[b ,1] * [b ,1] -> [b ,1] -> scala
        loss = (-probs.log() * delta).mean()

        #时序差分的loss就是简单的value和target求mse loss即可
        loss_td = loss_fn(values, targets.detach())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        optimizer_td.zero_grad()
        loss_td.backward()
        optimizer_td.step()

        if i % 100 == 0:
            test_result = sum([test(play=False) for _ in range(10)]) / 10
            print(i, test_result)


train()