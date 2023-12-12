import torch
from model import Model
# from env import MyWrapper
from model_test import test
from parallel_env.env_n import make_train_env
# 问题：环境没有被重置，env中的reset函数压根就没有被调用，导致收集到的数据有误，无法训练出效果，改掉后还是奖励不收敛
# test函数实例化了另一个模型，并未参与训练的，导致无法看出是否收敛


def main():
    model = Model()
    env = make_train_env(3)
    # env = MyWrapper()
    optimizer = torch.optim.Adam(model.Actor.parameters(), lr=1e-3)
    optimizer_td = torch.optim.Adam(model.Critic.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()

    #玩N局游戏,每局游戏训练一次
    for i in range(3000):
        #玩一局游戏,得到数据
        #states -> [b, 4]
        #rewards -> [b, 1]
        #actions -> [b, 1]
        #next_states -> [b, 4]
        #overs -> [b, 1]
        states, rewards, actions, next_states, overs = model.get_data(env)

        #计算values和targets
        #[b, 4] -> [b ,1]
        values = model.Critic(states)

        #[b, 4] -> [b ,1]
        targets = model.Critic(next_states) * 0.98
        #[b ,1] * [b ,1] -> [b ,1]
        targets *= (1 - overs)
        #[b ,1] + [b ,1] -> [b ,1]
        targets += rewards

        #时序差分误差
        #[b ,1] - [b ,1] -> [b ,1]
        delta = (targets - values).detach()

        #重新计算对应动作的概率
        #[b, 4] -> [b ,2]
        probs = model.Actor(states)
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
            test_result = sum([test(model, env) for _ in range(10)]) / 10
            # model.save("./model", "./model")
            print(i, test_result)


if __name__ == "__main__":
    main()

