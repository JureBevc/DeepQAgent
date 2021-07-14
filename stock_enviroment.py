from numpy.core.fromnumeric import size
import yfinance as yf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time


class StockEnviroment:

    def __init__(self, ticker_data=[]):
        if ticker_data != []:
            self.price = ticker_data
        else:
            self.price = np.array(
                np.sin(2 * np.pi * 0.01 * np.linspace(0, 300, 300)) * 0.5 + 0.5)
        self.volume = np.zeros(len(self.price))
        self.ma1 = self.moving_average(self.price, 20)
        self.ma2 = self.moving_average(self.price, 50)
        self.index = 0
        self.buycount = 0
        self.buycost = 0
        self.done = True
        self.action_log = []
        self.action_colors = []
        self.reward = 0
        self.figure = plt.figure()
        self.graph = None

    def reset(self):
        self.done = False
        self.index = 0
        self.buycost = 0
        self.buycount = 0
        self.reward = 0
        self.action_log = []
        self.action_colors = []
        self.info = {"net": 0, "low": 0, "high": 0}
        return np.array(
            [#self.price[self.index],
             self.volume[self.index],
             self.ma1[self.index],
             self.ma2[self.index]])

    def step(self, action):
        if self.done:
            return

        action_color = [0, 0, 1]
        self.reward = 0
        if action == 0:  # Do nothing
            pass
        elif action == 1:  # Buy

            if self.buycount > 0:  # Sell
                dif = self.price[self.index] * self.buycount - self.buycost
                self.info["net"] += dif
                if self.info["low"] > dif:
                    self.info["low"] = dif
                if self.info["high"] < dif:
                    self.info["high"] = dif

                self.buycount = 0
                self.buycost = 0
                self.reward = dif if dif >= 0 else dif
                action_color = [1, 0, 0]
            else:  # Buy
                self.buycount += 1
                self.buycost += self.price[self.index]
                action_color = [0, 1, 0]

        else:
            print(f"INVALID ACTION {action}!!!")

        self.action_log.append(action)
        self.action_colors.append(action_color)

        self.index += 1
        if self.index >= len(self.price) - 1:
            done = True
        else:
            done = False

        return np.array([
            self.volume[self.index],
            self.ma1[self.index],
            self.ma2[self.index]]), self.reward, done, self.info

    def render(self):
        if self.graph:
            self.graph.remove()
            first_render = False
        else:
            first_render = True

        self.graph = plt.scatter(np.linspace(
            0, 1, self.index), self.price[:self.index], color=self.action_colors, s=2)
        plt.pause(0.0001)

    def moving_average(self, arr, n):
        a = np.ma.masked_array(arr, np.isnan(arr))
        ret = np.cumsum(a.filled(0))
        ret[n:] = ret[n:] - ret[:-n]
        counts = np.cumsum(~a.mask)
        counts[n:] = counts[n:] - counts[:-n]
        ret[~a.mask] /= counts[~a.mask]
        ret[a.mask] = np.nan
        # ret[a.mask] = 0
        return ret
