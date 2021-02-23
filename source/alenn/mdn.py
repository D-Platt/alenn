'''
A class implementing the MDN-based likelihood approximation.

Donovan Platt
Mathematical Institute, University of Oxford
Institute for New Economic Thinking at the Oxford Martin School

2020-09-10

Copyright (c) 2020, University of Oxford. 
All rights reserved.

Distributed under a BSD 3-Clause licence. 
See the accompanying LICENCE file for further details.
'''

#-----------------------------------------------------------------------------------------------------------------
# Dependencies
    
# Import Numerical Computation Libraries
import tensorflow as tf
import numpy as np

# Import General Mathematical Libraries
import random as rn
from scipy import stats

# Import Keras Modules
from keras.models import Model
from keras.layers import Input, Dense, GaussianNoise

# Import Keras Backend
from keras import backend as K

# Import Miscellaneous Libraries
from types import *

#-----------------------------------------------------------------------------------------------------------------
# Main Class Source

class MDNPosterior:
    '''
    A class implementing the MDN-based likelihood and posterior approximations.
    
    Attributes
    ----------
    num_lags : int
        Number of lags.
    num_mix : int
        Number of mixture components.
    num_neurons : int
        Number of nodes per hidden layer.
    num_layers : int
        Number of hidden layers.
    batch_size : int
        Training batch size.
    num_epochs : int
        Number of training epochs.
    act_func : string
        A Keras activation function string.
    eta_x : float
        Noise regularisation standard deviation (network inputs).
    eta_y : float
        Noise regularisation standard deviation (mixture model output).
    model : function
        Simulator function for the candidate model.
    num_param : int
        The total number of free model parameters.
    priors : list
        A set priors over each free parameter.
    x_empirical : 2-d numpy array
        A set of MDN input training windows of length num_lags drawn from a set 
        of empirical data.
    y_empirical : 1-d numpy array
        The empirical observations directly following each window in x_empirical.
        
    Methods
    -------
    set_model(self, model)
        Set the model to be calibrated.
    set_prior(self, priors)
        Set the prior distribution for the Bayesian estimation procedure.
    load_data(self, empirical)
        Load the empirical data to which the candidate model is to be calibrated.
    evaluate_posterior(self, theta)
        Evaluate the log-posterior and log-likelihood for a given parameter set.
    '''
    
    #-------------------------------------------------------------------------------------------------------------
    # Class Constructor

    def __init__(self, num_lags = 3, num_mix = 16, num_neurons = 32, num_layers = 3, batch_size = 512, 
                 num_epochs = 12, act_func = 'relu', eta_x = 0.2, eta_y = 0.2):
        '''
        Class constructor method.
        
        Parameters
        ----------
        num_lags : int
            Number of lags.
        num_mix : int
            Number of mixture components.
        num_neurons : int
            Number of nodes per hidden layer.
        num_layers : int
            Number of hidden layers.
        batch_size : int
            Training batch size.
        num_epochs : int
            Number of training epochs.
        act_func : string
            A Keras activation function string.
        eta_x : float
            Noise regularisation standard deviation (network inputs).
        eta_y : float
            Noise regularisation standard deviation (mixture model output).
        
        Returns
        -------
        None.
        
        Calling the class constructor with no arguments simply loads the default 
        network configuration employed in the paper.
        '''
        
        # Set MDN Attributes
        self.num_lags = num_lags
        self.num_mix = num_mix
        self.num_neurons = num_neurons
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.act_func = act_func
        self.eta_x = eta_x
        self.eta_y = eta_y
        
        # Display an Object Creation Message to the User
        print('--------------------------------------')
        print('Successfully created a new MDN object:')
        print('--------------------------------------')

        print('{: <30} {: <20}'.format('Number of lags:', num_lags))
        print('{: <30} {: <20}'.format('Number of mixture components:', num_mix))
        print('{: <30} {: <20}'.format('Number of neurons per layer:', num_neurons))
        print('{: <30} {: <20}'.format('Number of hidden layers:', num_layers))
        print('{: <30} {: <20}'.format('Batch size:', batch_size))
        print('{: <30} {: <20}'.format('Number of epochs:', num_epochs))
        print('{: <30} {: <20}'.format('Activation function:', act_func))
        print('{: <30} {: <20}'.format('Input noise:', eta_x))
        print('{: <30} {: <20}'.format('Output noise:', eta_y))
        
        print('--------------------------------------')
        print('')
        
    #-------------------------------------------------------------------------------------------------------------
    # User-called Methods

    def set_model(self, model):
        '''
        Set the model to be calibrated.
        
        Parameters
        ----------
        model : function
            A Python function that should take in a single list argument
            containing model parameters and return a corresponding set of 
            R Monte Carlo replications of length T_sim as an T_sim x R numpy 
            array.

        Returns
        -------
        None.
        '''
        
        # Attempt to Create a New Attribute to Store the Provided Model Function
        if type(model) == FunctionType:
            self.model = model
            print('Model function successfully set.')
            print('----------------------------------------------------------------------------')
            print('')
        else:
            print('Error: Provided argument is not a function.')
                 
    def set_prior(self, priors):
        '''
        Set the prior distribution for the Bayesian estimation procedure.
        
        Parameters
        ----------
        priors : list
            A list of functions of length equal to the number of free parameters
            in the candidate model. Each function should take in a single real-valued
            argument, the current value for the parameter, and return a single, 
            corresponding real-valued output, the prior probability.

        Returns
        -------
        None.
        '''
        
        # Attempt to Create a New Attribute to Store the Provided Prior Functions
        if type(priors) == list:
            if np.array([type(priors[i]) == FunctionType or type(priors[i]) == MethodType for i in range(len(priors))]).prod():
                self.priors = priors
                self.num_param = len(priors)
                print('Model prior successfully set. The model has %d free parameters.' % self.num_param)
                print('----------------------------------------------------------------------------')
                print('')
            else:
                print('Error: Provided argument is not a list of functions.')
        else:
            print('Error: Provided argument is not a list of functions.')
        
    def load_data(self, empirical):
        '''
        Load the empirical data to which the candidate model is to be calibrated.
        
        Parameters
        ----------
        empirical : 1-d numpy array
            A set of empirical data to which the model is to be calibrated.

        Returns
        -------
        None
        '''
        
        # Attempt to Create New Attributes to Store the Provided Empirical Data
        try:
            
            # Check that the Data is 1-d
            if len(empirical.shape) == 1:
                
                # Determine the Data Size
                T_emp = len(empirical)
                
                # Convert to Rolling Window Sets
                self.x_empirical = self.__state_extraction_past(empirical.reshape(1, T_emp), self.num_lags)
                self.y_empirical = self.__state_extraction_current(empirical.reshape(1, T_emp), self.num_lags)
                
                # Display a Success Message
                print('Empirical data successfully loaded. There are %d observations in total.' % T_emp)
                print('----------------------------------------------------------------------------')
                print('')
                      
            else:
                
                # Display a Failure Message
                print('Error: Provided data is not a 1-d numpy array.')
         
        except:
            
            # Display a Failure Message
            print('Error: Provided data is not a 1-d numpy array.')
    
    def evaluate_posterior(self, theta):
        '''
        Evaluate the log-posterior and log-likelihood for a given parameter set.
        
        Parameters
        ----------
        theta : 1-d numpy array.
            An array of model parameters at which to evaluate the posterior.

        Returns
        -------
        output_list : list
            A python list containing, in order, the log-posterior, log p(X | theta) 
            + log p(theta), and the log-likelihood, log p(X | theta).
        '''
                
        # Evaluate Prior
        prior_product = np.array([self.priors[i](theta[i]) for i in range(len(self.priors))]).prod()
    
        # Check for Zero Prior
        if prior_product == 0:
            return [-np.inf, np.nan]
        
        # Generate Training Data
        x_train_raw = np.transpose(self.model(theta))
        x_train = self.__state_extraction_past(x_train_raw, self.num_lags)
        y_train = self.__state_extraction_current(x_train_raw, self.num_lags)
        
        # Determine Normalisation Parameters
        x_mean = x_train.mean(axis = 0)
        x_std = x_train.std(axis = 0)
        y_mean = y_train.mean(axis = 0)
        y_std = y_train.std(axis = 0)

        # Normalise Training Data
        x_train = (x_train - x_mean) / x_std
        y_train = (y_train - y_mean) / y_std
                
        # Initialise Neural Network Computational Graph
        self.__initialise_network_graph()
        
        # Define Target Data Layer
        y = Input(shape = (1,))

        # Define Neural Network Input Layer
        x = Input(shape = (self.num_lags,))

        # Apply Noise Regularisation
        x_reg = GaussianNoise(self.eta_x)(x)
        y_reg = GaussianNoise(self.eta_y)(y)
        
        # Define Neural Network Hidden Layers
        h = Dense(self.num_neurons, activation = self.act_func)(x_reg)
        
        for i in range(1, self.num_layers):
            h = Dense(self.num_neurons, activation = self.act_func)(h)
        
        # Define Neural Network Output Layers
        alpha = Dense(self.num_mix, activation = 'softmax')(h)
        mean = Dense(self.num_mix)(h)
        log_var = Dense(self.num_mix)(h)
        
        # Instantiate Mixture Density Network
        mdn = Model([x, y], [alpha, mean, log_var])

        # Add Loss Function
        mdn.add_loss(self.__loss_func(y_reg, self.num_mix, alpha, mean, log_var))

        # Compile Mixture Density Network
        mdn.compile(optimizer = 'adam')

        # Fit Mixture Density Network
        mdn.fit([x_train, y_train],
                epochs = self.num_epochs,
                batch_size = self.batch_size,
                verbose = False)
    
        # Create Data Structure to Store Likelihood Values
        likelihood = np.zeros(len(self.y_empirical))
        
        # Determine Likelihood Values
        for i in range(len(self.y_empirical)):
        
            # Determine Mixture Parameters
            alpha, mean, log_var = mdn.predict([(self.x_empirical[i, :].reshape(1, self.num_lags) - x_mean) / x_std, np.array([0])])
            
            # Determine Likelihood Value
            likelihood[i] = (1 / y_std.prod()) * self.__gmm_univar((self.y_empirical[i] - y_mean) / y_std, self.num_mix, alpha.flatten(), mean.flatten(), log_var.flatten())
            
        # Terminate Neural Network Computational Graph
        self.__terminate_network_graph()
        
        # Calculate the Log-likelihood
        ll = np.log(likelihood).sum()
                
        # Check for NaNs
        if np.isnan(ll):
            return [-np.inf, np.nan]
        
        # Return the Log-posterior and Log-likelihood
        return [ll + np.log(prior_product), ll]
    
    #-------------------------------------------------------------------------------------------------------------
    # Non-user Helper Methods

    def __state_extraction_past(self, x, l):
        '''
        Transforms x into a set of rolling windows of length l.
        '''
        return np.array([x[i, j : j + l] for i in range(x.shape[0]) for j in range(x.shape[1] - l)])

    def __state_extraction_current(self, x, l):
        '''
        Returns the observations in x that directly follow the rolling windows 
        of length l generated by state_extraction_past.
        '''
        return np.array([x[i, j] for i in range(x.shape[0]) for j in range(l, x.shape[1])])
    
    def __gmm_univar(self, y, k, alpha, m, log_var):
        '''
        Return the density of point y, assuming a Gaussian mixture model with 
        mixture weights alpha, mean vector m, and log variance vector log_var.
        '''
        
        # Convert the Log Variance to the Standard Deviation
        s = np.exp(0.5 * log_var)
        
        # Evaluate the Density Function
        return (np.array([stats.norm.pdf(y, loc = m[i], scale = s[i]) for i in range(k)]) * alpha).sum()

    def __loss_func(self, y, k, alpha, m, log_var):
        '''
        Maximum likelihood-based Loss function used to train the MDN.
        '''
    
        # Convert the Log Variance to the Standard Deviation
        s = K.exp(0.5 * log_var)
        
        # Calculate the Coefficient for the Exponential Function
        coeff = (2 * np.pi) ** (-0.5)
        coeff = coeff / s
        
        # Determine the Exponent for the Exponential Function
        exponent = -K.square(y - m) / (2 * K.square(s))
        
        # Return the Loss Function
        return -K.sum(K.log(K.sum(alpha * coeff * K.exp(exponent), axis = 1)), axis = 0)
    
    def __initialise_network_graph(self):
        '''
        Initialises the Tensorflow graph associated with the current MDN model.
        '''
        
        # Initialise Graph
        sess = tf.Session(graph = tf.get_default_graph())
        K.set_session(sess)
        
        # Seed Random Number Generators (j*)
        np.random.seed(12)
        rn.seed(13)
        tf.set_random_seed(14)
        
    def __terminate_network_graph(self):
        '''
        Terminates the Tensorflow graph associated with the current MDN model.
        '''
        
        # Clear Graph
        K.clear_session()
        
        # Reset Random Number Generators
        np.random.seed()
        rn.seed()