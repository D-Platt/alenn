'''
A class implementing the KDE-based likelihood approximation, as introduced by 
Grazzini et al. (2017).

Donovan Platt
Mathematical Institute, University of Oxford
Institute for New Economic Thinking at the Oxford Martin School

2020-10-12

Copyright (c) 2020, University of Oxford. 
All rights reserved.

Distributed under a BSD 3-Clause licence. 
See the accompanying LICENCE file for further details.
'''

#-----------------------------------------------------------------------------------------------------------------
# Dependencies
    
# Import Numerical Computation Libraries
import numpy as np

# Import General Mathematical Libraries
from scipy import stats

# Import Miscellaneous Libraries
from types import *

#-----------------------------------------------------------------------------------------------------------------
# Main Class Source

class KDEPosterior:
    '''
    A class implementing the KDE-based likelihood and posterior approximations.
    
    Attributes
    ----------
    model : function
        Simulator function for the candidate model.
    num_param : int
        The total number of free model parameters.
    priors : list
        A set priors over each free parameter.
    empirical : 1-d numpy array
        The empirical data to which the model is to be calibrated.
        
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

    def __init__(self):
        '''
        Class constructor method.
        
        Parameters
        ----------
        None.
        
        Returns
        -------
        None.
        '''
        
        # Display an Object Creation Message to the User
        print('------------------------------------------')
        print('Successfully created a new KDE object:')
        print('------------------------------------------')
        print('')
        print('Using a Silverman bandwidth approximation.')
        print('------------------------------------------')
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
        
        # Attempt to Create a New Attribute to Store the Provided Empirical Data
        try:
            
            # Check that the Data is 1-d
            if len(empirical.shape) == 1:
                
                # Determine the Data Size
                T_emp = len(empirical)
                
                # Store the Empirical Data as an Attribute
                self.empirical = empirical
                
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
        x_train_raw = self.model(theta).flatten()
    
        # Fit KDE Model (Silverman Bandwidth)
        pdf = stats.gaussian_kde(x_train_raw, 'silverman')
    
        # Determine the Likelihood of each Element in the Empirical Data
        likelihood = pdf.evaluate(self.empirical)
            
        # Calculate the Log-likelihood
        ll = np.log(likelihood).sum()
                
        # Check for NaNs
        if np.isnan(ll):
            return [-np.inf, np.nan]
        
        # Return the Log-posterior and Log-likelihood
        return [ll + np.log(prior_product), ll]