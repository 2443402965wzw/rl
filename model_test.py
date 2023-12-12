
def test(model, env):
    #初始化游戏
    state, info = env.reset()
    #记录反馈值的和,这个值越大越好
    reward_sum = 0

    #玩到游戏结束为止
    over = [False, False, False]
    while not any(over):
        #根据当前状态得到一个动作
        action = model.get_action(state)

        #执行动作,得到反馈
        state, reward, over, _, _ = env.step(action)
        reward_sum += reward[0]

    return reward_sum

