import yfinance as yf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

class StockEnviroment:

    def __init__(self):
        self.close = np.array(np.sin(2 * np.pi * 0.01 * np.linspace(0,300, 300)) * 0.5 + 0.5)
        self.volume = np.zeros(len(self.close))
        self.ma1 = self.moving_average(self.close, 20)
        self.ma2 = self.moving_average(self.close, 50)
        self.index = 0
        self.cost = 0
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
        self.cost = 0
        self.buycost = 0
        self.buycount = 0
        self.reward = 0
        self.action_log = []
        self.action_colors = []
        return np.array(
            [self.close[self.index],
            self.volume[self.index],
            self.ma1[self.index],
            self.ma2[self.index],
            0 if self.buycount == 0 else self.buycost / self.buycount])

    def step(self, action):
        if self.done:
            return

        action_color = [0,0,1]
        self.reward = 0
        if action == 0: # Do nothing
            pass
        elif action == 1: # Buy
            self.buycount += 1
            self.buycost += self.close[self.index]
            self.cost -= self.close[self.index]
            action_color = [0,1,0]
        elif action == 2: # Sell
            if self.buycount > 0:
                dif = self.close[self.index] * self.buycount - self.buycost
                self.cost += self.close[self.index] * self.buycount
                self.buycount = 0
                self.buycost = 0
                self.reward = dif if dif >= 0 else dif * 2
            else:
                self.reward = -100
            action_color = [1,0,0]
        else:
            print(f"INVALID ACTION {action}!!!")
        

        self.action_log.append(action)
        self.action_colors.append(action_color)


        self.index += 1
        if self.index >= len(self.close) - 1:
            done = True
        else:
            done = False
        
        return np.array([
            self.close[self.index],
            self.volume[self.index],
            self.ma1[self.index],
            self.ma2[self.index],
            0 if self.buycount == 0 else self.buycost / self.buycount]), self.reward, done, None


    def render(self):
        if self.graph:
            self.graph.remove()
            first_render = False
        else:
            first_render = True

        self.graph = plt.scatter(np.linspace(0,1,self.index), self.close[:self.index], color=self.action_colors)
        plt.pause(0.0001)

    def moving_average(self, arr, n):
        a = np.ma.masked_array(arr,np.isnan(arr))
        ret = np.cumsum(a.filled(0))
        ret[n:] = ret[n:] - ret[:-n]
        counts = np.cumsum(~a.mask)
        counts[n:] = counts[n:] - counts[:-n]
        ret[~a.mask] /= counts[~a.mask]
        ret[a.mask] = np.nan
        #ret[a.mask] = 0
        return ret