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
    global ticker, agent, env, plot_data, scores, run_random_agent
    done = False
    env_state = env.reset()
    while not done:
        env_state = env_state.astype(np.float32)
        action = agent.choose_action(env_state)
        new_env_state, reward, done, info = env.step(action)
        agent.store_transition(
            env_state, action, reward, new_env_state, done)
        # agent.learn()
        env_state = new_env_state
        plot_data.append({"net" : info["net"], "high" : info["high"], "low" : info["low"]})

    print("Finished running model")


if __name__ == "__main__":

    ticker = "MSFT" if len(sys.argv) < 2 else sys.argv[1]
    run_random_agent = len(sys.argv) > 2 and sys.argv[2] == "RANDOM"
    # Create enviroment and agent
    env = StockEnviroment(get_ticker_data(ticker))

    if run_random_agent:
        agent = RandomAgent(n_actions=2)
    else:
        agent = DeepQAgent(gamma=0.99, epsilon=0, epsilon_min=0.001, eps_dec=1e-6, batch_size=64,
                        n_actions=2, lr=0.001, input_dims=3)
        agent.load_from_file(ticker + ".mdl")

    # Training parameters
    scores = []
    plot_data = []

    model_thread = threading.Thread(target=model_loop)
    model_thread.daemon = True
    model_thread.start()

    plot_info_data()