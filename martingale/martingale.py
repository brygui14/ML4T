""""""
from itertools import filterfalse

"""Assess a betting strategy.

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---

Student Name: Bryan Indelicato (replace with your name)
GT User ID: bindelicato3 (replace with your User ID)
GT ID: 904061622 (replace with your GT ID)
"""

import numpy as np
import matplotlib.pyplot as plt

win_prob = np.round((18 / 38), 6)  # 18 Black slots out of a total of 38

def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "bindelicato3"  # replace tb34 with your Georgia Tech username.

def gtid():
    """
    :return: The GT ID of the student
    :rtype: int
    """
    return 904061622  # replace with your GT ID number

def get_spin_result(win_prob):
    """
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.

    :param win_prob: The probability of winning
    :type win_prob: float
    :return: The result of the spin.
    :rtype: bool
    """
    result = False
    if np.random.random() <= win_prob:
        result = True
    return result

def rouletteEpisode(episode, successive_bets=1000, winnings_threshold=80, bet_amount=1, bankroll=100000):
    episode_winnings = 0
    count = 0
    flag = False

    while episode_winnings < winnings_threshold or count < successive_bets + 1:
        if flag == True:
            break
        count += 1
        won = False
        bet = bet_amount

        while not won:
            if count > successive_bets or episode_winnings > winnings_threshold or bankroll < 0:
                episode[count:] = episode_winnings
                flag = True
                break
            won = get_spin_result(win_prob)
            if won == True:
                episode_winnings = episode_winnings + bet
                bankroll += bet
                episode[count] = episode_winnings
            else:
                episode_winnings -= bet
                bankroll -= bet
                bet *= 2
                episode[count] = episode_winnings
                count += 1
    return episode

def fig1():
    fig = plt.figure()
    ax = fig.gca()

    episodes = np.zeros((10, 1001))

    for i in range(episodes.shape[0]):
        episodes[i] = rouletteEpisode(episodes[i])
        plt.plot(episodes[i], label=f"Episode: {str(i + 1)}")

    ax.set_xlim([0,300])
    ax.set_ylim([-256, 100])
    plt.xlabel("Bet #")
    plt.ylabel("Episode Winnings $")
    ax.set_title("Figure 1")
    plt.legend(loc=4)
    plt.savefig('images/figure_1.png')

def fig2():
    fig = plt.figure()
    ax = fig.gca()

    episodes = np.zeros((1003, 1001))

    min = episodes.shape[0] - 3

    for i in range(episodes.shape[0] - 3):
        episodes[i] = rouletteEpisode(episodes[i])

    episodes[min] = np.mean(episodes, axis=0)
    episodes[min + 1] = np.mean(episodes, axis=0) + np.std(episodes, axis=0)
    episodes[min + 2] = np.mean(episodes, axis=0) - np.std(episodes, axis=0)

    plt.plot(episodes[min], label="Mean")
    plt.plot(episodes[min + 1], label="Upper STD")
    plt.plot(episodes[min + 2], label="Lower STD")
    ax.set_xlim([0,300])
    ax.set_ylim([-256, 100])
    plt.xlabel("Bet #")
    plt.ylabel("Episode Winnings $")
    ax.set_title("Figure 2")
    plt.legend(loc=4)
    plt.savefig('images/figure_2.png')

    fig3(episodes)

def fig3(episodes):
    fig = plt.figure()
    ax = fig.gca()

    min = episodes.shape[0] - 3

    episodes[min] = np.median(episodes, axis=0)
    episodes[min + 1] = np.mean(episodes, axis=0) + np.std(episodes, axis=0)
    episodes[min + 2] = np.mean(episodes, axis=0) - np.std(episodes, axis=0)

    plt.plot(episodes[min], label="Median")
    plt.plot(episodes[min + 1], label="Upper STD")
    plt.plot(episodes[min + 2], label="Lower STD")
    ax.set_xlim([0,300])
    ax.set_ylim([-256, 100])
    plt.xlabel("Bet #")
    plt.ylabel("Episode Winnings $")
    ax.set_title("Figure 3")
    plt.legend(loc=4)
    plt.savefig('images/figure_3.png')

def fig4():
    fig = plt.figure()
    ax = fig.gca()

    episodes = np.zeros((1003, 1001))

    min = episodes.shape[0] - 3

    for i in range(episodes.shape[0] - 3):
        episodes[i] = rouletteEpisode(episodes[i], bankroll=256)

    episodes[min] = np.mean(episodes, axis=0)
    episodes[min + 1] = np.mean(episodes, axis=0) + np.std(episodes, axis=0)
    episodes[min + 2] = np.mean(episodes, axis=0) - np.std(episodes, axis=0)

    plt.plot(episodes[min], label="Mean")
    plt.plot(episodes[min + 1], label="Upper STD")
    plt.plot(episodes[min + 2], label="Lower STD")
    ax.set_xlim([0,300])
    ax.set_ylim([-256, 100])
    plt.xlabel("Bet #")
    plt.ylabel("Episode Winnings $")
    ax.set_title("Figure 4")
    plt.legend(loc=4)
    plt.savefig('images/figure_4.png')

def fig5():
    fig = plt.figure()
    ax = fig.gca()

    episodes = np.zeros((1003, 1001))

    min = episodes.shape[0] - 3

    for i in range(episodes.shape[0] - 3):
        episodes[i] = rouletteEpisode(episodes[i], bankroll=256)

    episodes[min] = np.median(episodes, axis=0)
    episodes[min + 1] = np.mean(episodes, axis=0) + np.std(episodes, axis=0)
    episodes[min + 2] = np.mean(episodes, axis=0) - np.std(episodes, axis=0)

    plt.plot(episodes[min], label="Median")
    plt.plot(episodes[min + 1], label="Upper STD")
    plt.plot(episodes[min + 2], label="Lower STD")
    ax.set_xlim([0,300])
    ax.set_ylim([-256, 100])
    plt.xlabel("Bet #")
    plt.ylabel("Episode Winnings $")
    ax.set_title("Figure 5")
    plt.legend(loc=4)
    plt.savefig('images/figure_5.png')

def test_code():
    np.random.seed(gtid())  # do this only once

    fig1()
    fig2()
    fig4()
    fig5()

if __name__ == "__main__":
    test_code()
