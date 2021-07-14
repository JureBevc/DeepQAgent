from random_agent import RandomAgent
import time
import threading
import torch
import sys
import yfinance as yf
from pynput import keyboard
from time import sleep
from q_agent import DeepQAgent
from stock_enviroment import StockEnviroment
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style


def plot_info_data():
    global ticker, plot_data
    fig = plt.figure()

    def plot_loop(i):
        net_data = [info["net"] for info in plot_data]
        high_data = [info["high"] for info in plot_data]
        low_data = [info["low"] for info in plot_data]
        plt.clf()
        plt.title(ticker)
        plt.plot(net_data, "-", label="Net")
        plt.plot(high_data, "--", label="High")
        plt.plot(low_data, "--", label="Low")
    
    style.use("fivethirtyeight")
    ani = animation.FuncAnimation(fig, plot_loop, interval=1000)
    plt.show()

def get_ticker_data(ticker):
    data = yf.download(
        tickers=ticker,
        period="2y",
        interval="1d",
    )
    values = data["Close"].values
    print(f"Downloaded {ticker} data with {len(values)} entries")
    return values


def model_loop():
    global ticker, agent, env, max_games, plot_data, scores, run_random_agent
    for i in range(max_games):
        score = 0
        done = False
        env_state = env.reset()
        while not done:
            env_state = env_state.astype(np.float32)
            action = agent.choose_action(env_state)
            new_env_state, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(
                env_state, action, reward, new_env_state, done)
            agent.learn()
            env_state = new_env_state
        scores.append(score)
        avg_score = np.mean(scores[-100:])

        plot_data.append(env.info)
        print(
            f"Game {i}: Score {round(score,2)} | Average score {round(avg_score, 2)}")
        print(
            f"\tTrade summary: Net {round(env.info['net'],2)} | Low {round(env.info['low'],2)} | High: {round(env.info['high'],2)}")
        if i % 10 == 0 and run_random_agent == False:
            # Save model
            torch.save(agent.target_model.state_dict(), ticker + ".mdl")
        if run_random_agent:
            time.sleep(0.01)
        else:
            print(agent.epsilon)

if __name__ == "__main__":

    ticker = "MSFT" if len(sys.argv) < 2 else sys.argv[1]
    run_random_agent = len(sys.argv) > 2 and sys.argv[2] == "RANDOM"
    # Create enviroment and agent
    env = StockEnviroment(get_ticker_data(ticker))

    if run_random_agent:
        agent = RandomAgent(n_actions=2)
    else:
        agent = DeepQAgent(gamma=0.99, epsilon=1.0, epsilon_min=0.001, eps_dec=1e-6, batch_size=64,
                        n_actions=2, lr=0.001, input_dims=3)

    # Training parameters
    scores = []
    max_games = 50000000
    plot_data = []

    model_thread = threading.Thread(target=model_loop)
    model_thread.daemon = True
    model_thread.start()

    plot_info_data()