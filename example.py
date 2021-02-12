import gym
from q_agent import DeepQAgent
from stock_enviroment import StockEnviroment
import numpy as np
import matplotlib.pyplot as plt
from pynput import keyboard

show_graph = False

def on_press(key):
    global show_graph
    try:
        if key.char == 'g':
            show_graph = not show_graph
    except Exception as e:
        pass

if __name__ == "__main__":
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    #env = gym.make("CartPole-v1")
    env = StockEnviroment()
    agent = DeepQAgent(gamma=0.99, epsilon=1.0, epsilon_min=0.001, eps_dec=5e-5, batch_size=64,
    n_actions=3, lr=0.001, input_dims=5)
    scores, eps = [], []
    max_games = 50000000

    for i in range(max_games):
        score = 0
        done = False
        env_state = env.reset()
        while not done:
            env_state = env_state.astype(np.float32)
            action = agent.choose_action(env_state)
            new_env_state, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(env_state, action, reward, new_env_state, done)
            agent.learn()
            env_state = new_env_state
            if show_graph:
                env.render()
        scores.append(score)
        eps.append(agent.epsilon)
        avg_score = np.mean(scores[-100:])

        print(f"Game {i}: Score {score} | Average score {avg_score} | Epsilon {agent.epsilon}")

    # Plot
    scores = np.array(scores)
    scores /= np.max(np.abs(scores))
    plt.plot(scores)
    plt.plot(eps)
    plt.show()

    listener.join()