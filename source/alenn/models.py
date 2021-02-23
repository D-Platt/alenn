'''
A set of functions implementing a suite of test models for ABM estimation.

Donovan Platt
Mathematical Institute, University of Oxford
Institute for New Economic Thinking at the Oxford Martin School

2020-09-18

Copyright (c) 2020, University of Oxford. 
All rights reserved.

Distributed under a BSD 3-Clause licence. 
See the accompanying LICENCE file for further details.
'''

#-----------------------------------------------------------------------------------------------------------------
# Dependencies
    
# Import Numerical Computation Libraries
import numpy as np

#-----------------------------------------------------------------------------------------------------------------
# Model Functions

def brock_hommes(g_1, b_1, g_2, b_2, g_3, b_3, g_4, b_4, r, beta, sigma, p_star, T, n, seed):
    '''
    Generate Brock and Hommes (1998) model realisations.
    
    Parameters
    ----------
    g_1 : float
        Trend following component for strategy 1.
    b_1 : float
        Bias for strategy 1.
    g_2 : float
        Trend following component for strategy 2.
    b_2 : float
        Bias for strategy 2.
    g_3 : float
        Trend following component for strategy 3.
    b_3 : float
        Bias for strategy 3.
    g_4 : float
        Trend following component for strategy 4.
    b_4 : float
        Bias for strategy 4.
    r : float
        Prevailing interest rate.
    beta : float
        Intensity of choice for agent strategy switching.
    sigma : float
        Noise term standard deviation.
    p_star : float
        Fundamental value.
    T : int
        Simulation length.
    n : int
        Number of Monte Carlo replications.
    seed : int
        Seed for the random number generators.
        
    Returns
    -------
    output_array : 2-d numpy array
        A T x n numpy array containing the generated model Monte Carlo replications.
    '''
    
    # Set Random Number Generator Seed
    np.random.seed(seed)
    
    # Note that if n > 1, setting the seed as above is equivalent to assigning 
    # each Monte Carlo replication its own seed.
    
    # Calculate Accumulation Factor
    R = 1 + r
    
    # Create Parameter Arrays
    g = np.array([g_1, g_2, g_3, g_4])
    b = np.array([b_1, b_2, b_3, b_4])
    
    # Create Data Structure to Store Price Process (Deviation from Fundamental Value)
    x = np.zeros((T, n))
        
    # Generate Random Shocks
    epsilon = np.random.normal(size = (T, n), scale = sigma)
 
    # Iterate Model (Time Domain)
    for i in range(T - 3):
        
        # Iterate Model (Monte Carlo Replications)
        for j in range(n):
            
            # Calculate Profitability
            U_h = (x[i + 2, j] - R * x[i + 1, j]) * (g * x[i, j] + b - R * x[i + 1, j])
            
            # Calculate Strategy Percentages
            n_h = np.exp(beta * U_h) / np.exp(beta * U_h).sum()
            
            # Update Price Process
            x[i + 3, j] = ((n_h * (g * x[i + 2, j] + b)).sum() + epsilon[i + 3, j]) / R
    
    # Reset Random Number Generators
    np.random.seed()
    
    # Account for the Fundamental Value and Return the Generated Result
    return x + p_star

def random_walk(t_b, d_1, d_2, sigma_1, sigma_2, T, n, seed):
    '''
    Generate model realisations for a random walk with a structural break.
    
    Parameters
    ----------
    t_b : int
        Structural break location.
    d_1 : float
        Pre-break drift.
    d_2 : float
        Post-break drift.
    sigma_1 : float
        Pre-break standard deviation.
    sigma_2 : float
        post-break standard deviation.
    T : int
        Simulation length.
    n : int
        Number of Monte Carlo replications.
    seed : int
        Seed for the random number generators.

    Returns
    -------
    output_array : 2-d numpy array
        A T x n numpy array containing the generated model Monte Carlo replications.
    '''
    
     # Set Random Number Generator Seed
    np.random.seed(seed)
    
    # Note that if n > 1, setting the seed as above is equivalent to assigning 
    # each Monte Carlo replication its own seed.
    
    # Create Data Structures to Store Processes
    x = np.zeros((T, n))
    d = np.zeros((T, 1))
    sigma = np.zeros((T, 1))
    
    # Determine Pre- and Post-Break Indices
    pre_break = np.arange(T) <= t_b
    post_break = np.arange(T) > t_b
    
    # Populate Drift and Volatility According to Break Point
    sigma[pre_break] = sigma_1
    sigma[post_break] = sigma_2
    d[pre_break] = d_1
    d[post_break] = d_2
    
    # Generate Random Shocks
    epsilon = np.random.normal(scale = sigma, size = (T, n))
    
    # Iterate Model (Time Domain)
    for i in range(T - 1):
        
        # Iterate Model (Monte Carlo Replications)
        for j in range(n):
            x[i + 1, j] = x[i, j] + d[i + 1] + epsilon[i + 1, j]
    
    # Reset Random Number Generators
    np.random.seed()
    
    # Return Generated Result
    return x

def fw_hpm(mu, beta, phi, chi, alpha_0, alpha_n, alpha_p, sigma_f, sigma_c, p_star, T, n, seed):
    '''
    Generate realisations of the DCA-HPM version of the Franke and Westerhoff (2012)
    model.
    
    Parameters
    ----------
    mu : float
        Aggregate demand scaling factor.
    beta : float
        Intensity of choice for agent strategy switching.
    phi : float
        Sensitivity of fundamentalist demand to current levels of mid-pricing.
    chi : float
        Sensitivity of chartist demand to past trends.
    alpha_0 : float
        Degree of predisposition of trader agents to fundamentalism.
    alpha_n : float
        Strength of agent herding tendencies.
    alpha_p : float
        Attractiveness of fundamentalism due to current levels of mispricing.
    sigma_f : float
        Fundamentalist demand standard deviation.
    sigma_c : float
        Chartist demand standard deviation.
    p_star : float
        Log-fundamental value.
    T : int
        Simulation length.
    n : int
        Number of Monte Carlo replications.
    seed : int
        Seed for the random number generators.

    Returns
    -------
    output_array : 2-d numpy array
        A T x n numpy array containing the generated model Monte Carlo replications.
    '''
    
    # Set Random Number Generator Seed
    np.random.seed(seed)
    
    # Note that if n > 1, setting the seed as above is equivalent to assigning 
    # each Monte Carlo replication its own seed.
    
    # Create Data Structures to Store the Various Time Series Generated by the Model
    p = np.zeros([T + 1, n])
    n_f = np.zeros([T, n])
    d_f = np.zeros([T, n])
    d_c = np.zeros([T, n])
    a = np.zeros([T, n])
    
    # Initialise the Population
    n_f[0 : 2, :] = 0.5
    
    # Iterate the Model for the Desired Number of Iterations
    for t in range(2, T):
        
        # Update the Agent Population Variable
        n_f[t, :] = 1 / (1 + np.exp(-beta * a[t - 1, :]))
        
        # Update the Relative Attractiveness
        a[t, :] = alpha_n * (2 * n_f[t, :] - 1) + alpha_0 + alpha_p * (p[t, :] - p_star) ** 2
        
        # Update the Net Demands
        d_f[t, :] = phi * (p_star - p[t, :]) + np.random.normal(loc = 0, scale = sigma_f, size = (1, n))
        d_c[t, :] = chi * (p[t, :] - p[t - 1, :]) + np.random.normal(loc = 0, scale = sigma_c, size = (1, n))    
        
        # Update the Price
        p[t + 1, :] = p[t , :] + mu * (n_f[t, :] * d_f[t, :] + (1 - n_f[t, :]) * d_c[t, :])    
    
    # Reset Random Number Generators
    np.random.seed()
    
    # Return the Price Time Series
    return p

def fw_wp(mu, beta, phi, chi, alpha_0, alpha_w, eta, sigma_f, sigma_c, p_star, T, n, seed):
    '''
    Generate realisations of the DCA-WP version of the Franke and Westerhoff (2012)
    model.
    
    Parameters
    ----------
    mu : float
        Aggregate demand scaling factor.
    beta : float
        Intensity of choice for agent strategy switching.
    phi : float
        Sensitivity of fundamentalist demand to current levels of mid-pricing.
    chi : float
        Sensitivity of chartist demand to past trends.
    alpha_0 : float
        Degree of predisposition of trader agents to fundamentalism.
    alpha_w : float
        Strength of agent wealth-chasing behavior.
    eta : float
        Wealth process memory.
    sigma_f : float
        Fundamentalist demand standard deviation.
    sigma_c : float
        Chartist demand standard deviation.
    p_star : float
        Log-fundamental value.
    T : int
        Simulation length.
    n : int
        Number of Monte Carlo replications.
    seed : int
        Seed for the random number generators.

    Returns
    -------
    output_array : 2-d numpy array
        A T x n numpy array containing the generated model Monte Carlo replications.
    '''
    
    # Set Random Number Generator Seed
    np.random.seed(seed)
    
    # Note that if n > 1, setting the seed as above is equivalent to assigning 
    # each Monte Carlo replication its own seed.
    
    # Create Data Structures to Store the Various Time Series Generated by the Model
    p = np.zeros([T + 1, n])
    d_f = np.zeros([T, n])
    d_c = np.zeros([T, n])
    n_f = np.zeros([T, n])
    g_f = np.zeros([T, n])
    g_c = np.zeros([T, n])
    w_f = np.zeros([T, n])
    w_c = np.zeros([T, n])
    a = np.zeros([T, n])
    
    # Initialise the Population
    n_f[0 : 2, :] = 0.5
    
    # Iterate the Model for the Desired Number of Iterations
    for t in range(2, T):
        
        # Update Current Portfolio Performance
        g_f[t, :] = (np.exp(p[t, :]) - np.exp(p[t - 1, :])) * d_f[t - 2, :]
        g_c[t, :] = (np.exp(p[t, :]) - np.exp(p[t - 1, :])) * d_c[t - 2, :]
    
        # Update Agent Wealth
        w_f[t, :] = eta * w_f[t - 1, :] + (1 - eta) * g_f[t, :]
        w_c[t, :] = eta * w_c[t - 1, :] + (1 - eta) * g_c[t, :]

        # Update the Agent Population Variable
        n_f[t, :] = 1 / (1 + np.exp(-beta * a[t - 1, :]))
        
        # Update the Relative Attractiveness
        a[t, :] = alpha_w * (w_f[t, :] - w_c[t, :]) + alpha_0
        
        # Update the Net Demands
        d_f[t, :] = phi * (p_star - p[t, :]) + np.random.normal(loc = 0, scale = sigma_f, size = (1, n))
        d_c[t, :] = chi * (p[t, :] - p[t - 1, :]) + np.random.normal(loc = 0, scale = sigma_c, size = (1, n))    
        
        # Update the Price
        p[t + 1, :] = p[t , :] + mu * (n_f[t, :] * d_f[t, :] + (1 - n_f[t, :]) * d_c[t, :])    
    
    # Reset Random Number Generators
    np.random.seed()
    
    # Return the Price Time Series
    return p

def ar_garch(a_0, a_1, a_2, omega, alpha_1, beta_1, T, n, seed):
    '''
    Generate AR-GARCH model realisations.
    
    Parameters
    ----------
    a_0 : float
        y-intercept.
    a_1 : float
        Lag 1 AR component.
    a_2 : float
        Lag 2 AR component.
    omega : float
        Minimum error variance.
    alpha_1 : float
        Lag 1 ARCH component.
    beta_1 : float
        Lag 1 GARCH component.
    T : int
        Simulation length.
    n : int
        Number of Monte Carlo replications.
    seed : int
        Seed for the random number generators.

    Returns
    -------
    output_array : 2-d numpy array
        A T x n numpy array containing the generated model Monte Carlo replications.
    '''
    
    # Set Random Number Generator Seed
    np.random.seed(seed)
    
    # Note that if n > 1, setting the seed as above is equivalent to assigning 
    # each Monte Carlo replication its own seed.
    
    # Create Data Structures to Store Processes
    x = np.zeros((T, n))
    sigma = np.zeros((T, n))
        
    # Generate Random Shocks
    z = np.random.normal(size = (T, n))
    
    # Set Initial Volatility
    sigma[0, :] = omega

    # Perform First GARCH Iteration
    sigma[1, :] = omega + alpha_1 * (z[0, :] ** 2) * sigma[0, :]  + beta_1 * sigma[0, :]
    
    # Iterate Model (Time Domain)
    for i in range(1, T - 1):
        
        # Iterate Model (Monte Carlo Replications)
        for j in range(n):
            
            # Update GARCH
            sigma[i + 1, j] = omega + alpha_1 * (z[i, j] ** 2) * sigma[i, j] + beta_1 * sigma[i, j]

            # Update ARMA
            x[i + 1, j] = a_0 + a_1 * x[i, j] + a_2 * x[i - 1, j] + np.sqrt(sigma[i + 1, j]) * z[i + 1, j]
        
    # Reset Random Number Generators
    np.random.seed()
    
    # Return Generated Result
    return x