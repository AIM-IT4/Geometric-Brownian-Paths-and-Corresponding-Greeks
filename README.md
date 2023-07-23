# Geometric-Brownian-Paths-and-Corresponding-Greeks

This Python code implements a financial option pricing model using the Black-Scholes (BS) formula and generates multiple stock price paths using the Geometric Brownian Motion (GBM) process. The code also plots various option-related surfaces using matplotlib.

The code begins by importing necessary libraries, including NumPy, matplotlib, and scipy.stats. It defines an enumeration class OptionType to represent the type of options (CALL or PUT). The main functions in the code are as follows:

GeneratePathsGBM: This function generates multiple stock price paths using the Geometric Brownian Motion (GBM) process. It takes parameters such as the number of paths (NoOfPaths), the number of time steps (NoOfSteps), the time to maturity (T), risk-free interest rate (r), volatility (sigma), and initial stock price (S_0).

BS_Call_Put_Option_Price: This function calculates the Black-Scholes option price for either a call or a put option. It takes parameters such as the option type (CP), initial stock price (S_0), strike price (K), volatility (sigma), time to maturity (t), maturity time (T), and risk-free interest rate (r).

BS_Delta: This function computes the Delta value for a call or put option. Delta measures the sensitivity of the option price to changes in the underlying stock price.

BS_Gamma: This function calculates the Gamma value for an option. Gamma represents the rate of change of Delta concerning changes in the underlying stock price.

BS_Vega: This function computes the Vega value for an option. Vega indicates the sensitivity of the option price to changes in volatility.

mainCalculation: This function is the main part of the code. It sets up parameters for the simulation and then generates stock price paths using GeneratePathsGBM. After that, it prepares lambda functions for option price calculations, and it calculates and stores the values of call and put options, Delta, Gamma, and Vega for different stock prices and times. Finally, it plots various option-related surfaces using matplotlib.

Overall, this code demonstrates how to use the Black-Scholes formula to price call and put options and provides insights into the behavior of option prices and Greeks (Delta, Gamma, and Vega) under different scenarios of stock prices and time to maturity.
