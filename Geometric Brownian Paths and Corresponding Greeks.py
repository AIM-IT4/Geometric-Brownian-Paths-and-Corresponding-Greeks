#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import enum
from mpl_toolkits import mplot3d
from scipy.interpolate import RegularGridInterpolator

class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0

def GeneratePathsGBM(NoOfPaths, NoOfSteps, T, r, sigma, S_0):
    Z = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])
    X = np.zeros([NoOfPaths, NoOfSteps + 1])
    W = np.zeros([NoOfPaths, NoOfSteps + 1])
    time = np.zeros([NoOfSteps + 1])

    X[:, 0] = np.log(S_0)
    dt = T / float(NoOfSteps)

    for i in range(0, NoOfSteps):
        if NoOfPaths > 1:
            Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])
        W[:, i + 1] = W[:, i] + np.power(dt, 0.5) * Z[:, i]
        X[:, i + 1] = X[:, i] + (r - 0.5 * sigma * sigma) * dt + sigma * (W[:, i + 1] - W[:, i])
        time[i + 1] = time[i] + dt

    S = np.exp(X)
    paths = {"time": time, "S": S}
    return paths

def BS_Call_Put_Option_Price(CP, S_0, K, sigma, t, T, r):
    K = np.array(K).reshape([len(K), 1])
    d1 = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma, 2.0)) * (T - t)) / (sigma * np.sqrt(T - t))
    d2 = d1 - sigma * np.sqrt(T - t)

    if CP == OptionType.CALL:
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * (T - t))
    elif CP == OptionType.PUT:
        value = st.norm.cdf(-d2) * K * np.exp(-r * (T - t)) - st.norm.cdf(-d1) * S_0

    return value

def BS_Delta(CP, S_0, K, sigma, t, T, r):
    K = np.array(K).reshape([len(K), 1])
    d1 = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma, 2.0)) * (T - t)) / (sigma * np.sqrt(T - t))

    if CP == OptionType.CALL:
        value = st.norm.cdf(d1)
    elif CP == OptionType.PUT:
        value = st.norm.cdf(d1) - 1

    return value

def BS_Gamma(S_0, K, sigma, t, T, r):
    K = np.array(K).reshape([len(K), 1])
    d1 = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma, 2.0)) * (T - t)) / (sigma * np.sqrt(T - t))
    return st.norm.pdf(d1) / (S_0 * sigma * np.sqrt(T - t))

def BS_Vega(S_0, K, sigma, t, T, r):
    d1 = (np.log(S_0 / K) + (r + 0.5 * np.power(sigma, 2.0)) * (T - t)) / (sigma * np.sqrt(T - t))
    return S_0 * st.norm.pdf(d1) * np.sqrt(T - t)

def mainCalculation():
    NoOfPaths = 500
    NoOfSteps = 50
    T = 1.001
    r = 0.05
    sigma = 0.4
    s0 = 10
    K = [10]
    pathId = 10

    np.random.seed(3)
    Paths = GeneratePathsGBM(NoOfPaths, NoOfSteps, T, r, sigma, s0)
    time = Paths["time"]
    S = Paths["S"]

    # Settings for the plots
    s0Grid = np.linspace(s0 / 100.0, 1.5 * s0, 50)
    timeGrid = np.linspace(0.02, T - 0.02, 100)

    # Prepare the necessary lambda functions
    CallOpt = lambda t, s0: BS_Call_Put_Option_Price(OptionType.CALL, s0, K, sigma, t, T, r)
    PutOpt = lambda t, s0: BS_Call_Put_Option_Price(OptionType.PUT, s0, K, sigma, t, T, r)
    DeltaCall = lambda t, s0: BS_Delta(OptionType.CALL, s0, K, sigma, t, T, r)
    Gamma = lambda t, s0: BS_Gamma(s0, K, sigma, t, T, r)
    Vega = lambda t, s0: BS_Vega(s0, K, sigma, t, T, r)

    # Prepare empty matrices for storing the results
    callOptM = np.zeros([len(timeGrid), len(s0Grid)])
    putOptM = np.zeros([len(timeGrid), len(s0Grid)])
    deltaCallM = np.zeros([len(timeGrid), len(s0Grid)])
    gammaM = np.zeros([len(timeGrid), len(s0Grid)])
    vegaM = np.zeros([len(timeGrid), len(s0Grid)])
    TM = np.zeros([len(timeGrid), len(s0Grid)])
    s0M = np.zeros([len(timeGrid), len(s0Grid)])

    for i in range(0, len(timeGrid)):
        TM[i, :] = timeGrid[i]
        s0M[i, :] = s0Grid
        callOptM[i, :] = CallOpt(timeGrid[i], s0Grid)
        putOptM[i, :] = PutOpt(timeGrid[i], s0Grid)
        deltaCallM[i, :] = DeltaCall(timeGrid[i], s0Grid)
        gammaM[i, :] = Gamma(timeGrid[i], s0Grid)
        vegaM[i, :] = Vega(timeGrid[i], s0Grid)

    # Plot stock path
    plt.figure(1)
    plt.plot(time, np.squeeze(S[pathId, :]))
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("S(t)")
    plt.plot(T, K, 'ok')

    # Plot the call option surface
    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(TM, s0M, callOptM, cmap='viridis')
    ax.set_xlabel('t')
    ax.set_ylabel('S(t)')
    ax.set_title('Call option surface')

    # Plot the put option surface
    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(TM, s0M, putOptM, cmap='viridis')
    ax.set_xlabel('t')
    ax.set_ylabel('S(t)')
    ax.set_title('Put option surface')

    # Plot the Delta for a call option surface
    fig = plt.figure(4)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(TM, s0M, deltaCallM, cmap='viridis')
    ax.set_xlabel('t')
    ax.set_ylabel('S(t)')
    ax.set_title('Delta for a call option surface')

    # Plot the Vega option surface
    fig = plt.figure(5)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(TM, s0M, vegaM, cmap='viridis')
    ax.set_xlabel('t')
    ax.set_ylabel('S(t)')
    ax.set_title('Vega surface')

    # Plot the Gamma option surface
    fig = plt.figure(6)
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(TM, s0M, gammaM, cmap='viridis')
    ax.set_xlabel('t')
    ax.set_ylabel('S(t)')
    ax.set_title('Gamma surface')

    # Show all the plots
    plt.show()

# Call the mainCalculation function when the script is run
if __name__ == "__main__":
    mainCalculation()

