import torch
import gym_2048
import DQN
import numpy as np
import logging
import logger
from torch.utils.tensorboard import SummaryWriter
import datetime
import tqdm


logger.logger(2)


def one_hot(state):
    one_hot_state = np.zeros((16, state.shape[0], state.shape[1]), dtype=np.float16)
    basecode = np.eye(16)
    for m in range(state.shape[0]):
        for n in range(state.shape[1]):
            value = state[m, n]
            one_hot_state[:, m, n] = basecode[int(np.log2(value) if value != 0 else 0), :]
    return one_hot_state


def log2_value(value, scale=15):
    return np.log2(1 + value) / scale


def train(filename=None, start_step=0, lr=1e-4, episodes=100001, epsilon_start=0.9, gamma=0.9):
    train_episodes = episodes
    env = gym_2048.Game2048Env()
    agent = DQN.DQN()
    agent.curr_step = start_step
    agent.learning_rate = lr
    agent.epsilon_start = epsilon_start
    agent.gamma = gamma
    if filename is not None:
        agent.policy_model = torch.load(filename)
        agent.target_model = torch.load(filename)
    ave_score = 0
    log_freq = 100
    log = dict()
    # log['highest'] = []
    # log['score'] = []
    # log['step'] = []
    # log['test_mean_score'] = []
    # log['test_mean_highest'] = []
    # log['test_max_score'] = []
    # log['test_max_highest'] = []
    # log['loss'] = []
    # log['test_loss'] = []
    writer.add_hparams({'lr': agent.learning_rate, 'batchsize': agent.batch_size},
                       {'epsilon': agent.epsilon, 'edecay': agent.epsilon_decay, 'emin': agent.epsilon_min,
                        'gamma': agent.gamma, 'gamma_up': agent.gamma_up})
    writer.add_graph(agent.policy_model, torch.rand(20, 16, 4, 4).to(agent.device))
    last_loss = 0
    for i in tqdm.trange(train_episodes):
        env.reset()
        # env.render()
        state = env.get_board()
        state = one_hot(state)
        loss_sum = 0
        learn_time = 0
        force_rand = False
        while True:
            # select action
            action = agent.select_action(state, force_rand=force_rand)
            # do action
            new_state, reward, done, info = env.step(action)
            new_state = one_hot(new_state)
            reward = log2_value(reward, 1) if reward > 0 else reward
            # reward = reward/128 if reward > 0 else reward
            # env.render()
            # remember
            agent.cache(state, action, new_state, reward, done)

            #
            if reward == -10:
                force_rand = True
            else:
                force_rand = False

            # update state
            state = new_state.copy()
            # learn
            loss = agent.learn()
            if loss is not None:
                last_loss = loss
            if done:
                ave_score += info['score']
                if i % log_freq == 0:
                    logging.info('--------------------')
                    logging.info('episodes:{}'.format(i))
                    logging.info('ave_score:{}'.format(ave_score / log_freq))
                    logging.info('epsilon:{}'.format(agent.epsilon))
                    writer.add_scalar(tag='score_train/ave_score', scalar_value=ave_score / log_freq, global_step=i)
                    writer.add_scalar(tag='tarin/epsilon', scalar_value=agent.epsilon, global_step=i)
                    writer.add_scalar(tag='tarin/gamma', scalar_value=agent.gamma, global_step=i)
                    ave_score = 0
                    # log['highest'].append(info['highest'])
                    # log['score'].append(info['score'])
                    # log['step'].append(info['steps'])
                    # torch.save(agent.policy_model,'dqn_{}.pkl'.format(info['highest']))
                    writer.add_scalar(tag='score_train/highest', scalar_value=info['highest'], global_step=i)
                    writer.add_scalar(tag='score_train/score', scalar_value=info['score'], global_step=i)
                    writer.add_scalar(tag='score_train/step', scalar_value=info['steps'], global_step=i)
                    writer.add_scalar(tag='tarin/loss', scalar_value=last_loss, global_step=agent.learn_step)
                    logging.info('loss:{}'.format(loss))
                break
        if i % 200 == 0:
            test_mean_score, test_mean_highest, test_max_score, test_max_highest = test(agent)
            # agent.learn()
            # log['test_mean_score'].append(test_mean_score)
            # log['test_mean_highest'].append(test_mean_highest)
            # log['test_max_score'].append(test_max_score)
            # log['test_max_highest'].append(test_max_highest)
            writer.add_scalar(tag='score_test/mean_highest', scalar_value=test_mean_highest, global_step=i)
            writer.add_scalar(tag='score_test/max_highest', scalar_value=test_max_highest, global_step=i)
            writer.add_scalar(tag='score_test/mean_score', scalar_value=test_mean_score, global_step=i)
            writer.add_scalar(tag='score_test/max_score', scalar_value=test_max_score, global_step=i)

    torch.save(agent.policy_model, f"dqn_final_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}.pkl")
    np.save('log_info.npy', log)
    writer.close()
    return log


def test(agent, test_episodes=10):
    env = gym_2048.Game2048Env()
    highest = []
    score = []
    for i in range(test_episodes):
        env.reset()
        state = env.get_board()
        state = one_hot(state)
        while True:
            action = agent.select_action(state, False)
            new_state, reward, done, info = env.step(action)
            new_state = one_hot(new_state)
            reward = log2_value(reward, 1) if reward > 0 else reward
            # reward = reward/128 if reward > 0 else reward
            # agent.cache(state,action,new_state,reward,done) # test过程无随机性，后期得分更高，考虑将test数据加入记忆
            state = new_state.copy()
            if done:
                highest.append(info['highest'])
                score.append(info['score'])
                break
    logging.info('*******************')
    test_mean_score, test_mean_highest, test_max_score, test_max_highest = np.sum(score) / test_episodes, np.sum(
        highest) / test_episodes, np.max(score), np.max(highest)
    logging.info('mean_score:{}'.format(test_mean_score))
    logging.info('mean_highest:{}'.format(test_mean_highest))
    logging.info('max_score:{}'.format(test_max_score))
    logging.info('max_highest:{}'.format(test_max_highest))
    logging.info('*******************')
    if test_max_highest>1024:
        torch.save(agent.policy_model, f"dqn_2048_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pkl")
    return test_mean_score, test_mean_highest, test_max_score, test_max_highest


def evaluate(filename, test_episodes):
    agent = DQN.DQN()
    agent.policy_model = torch.load(filename)
    env = gym_2048.Game2048Env()
    highest = []
    score = []
    for i in range(test_episodes):
        env.reset()
        state = env.get_board()
        state = one_hot(state)
        while True:
            action = agent.select_action(state, False)
            new_state, reward, done, info = env.step(action)
            new_state = one_hot(new_state)
            reward = log2_value(reward, 1) if reward > 0 else reward
            # agent.cache(state,action,new_state,reward,done) # test过程无随机性，后期得分更高，考虑将test数据加入记忆
            state = new_state.copy()
            if done:
                highest.append(info['highest'])
                score.append(info['score'])
                break
        print(f"highest:{info['highest']} score:{info['score']}")
    return highest,score


if __name__ == "__main__":

    # writer = SummaryWriter('runs3/{}'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    # writer = SummaryWriter('runs3/2022-11-23_22-02-42')
    # log = train('dqn_2048_2022-11-27_20-20-42.pkl', start_step=0, lr=1e-4, episodes=50000,epsilon_start=0.0001, gamma=0.96)
    # log = train()
    highest,score = evaluate(r'dqn_2048_2022-11-30_12-05-20.pkl',100)
    p = np.zeros(11)
    for k in range(len(p)):
        p[k] = highest.count(2**k)+p[k-1]
    p = 100-p
    print('1')
