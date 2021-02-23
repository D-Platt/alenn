'''
A class implementing the adaptive MCMC algorithm proposed by Griffin and Walker (2013).

Donovan Platt
Mathematical Institute, University of Oxford
Institute for New Economic Thinking at the Oxford Martin School

2020-10-08

Copyright (c) 2020, University of Oxford. 
All rights reserved.

Distributed under a BSD 3-Clause licence. 
See the accompanying LICENCE file for further details.
'''

#-----------------------------------------------------------------------------------------------------------------
# Dependencies
    
# Import Numerical Computation Libraries
import pandas as pd
import numpy as np

# Import General Mathematical Libraries
from scipy import stats

# Import Presentational Libraries
from tqdm import tnrange

# Import Custom Libraries
from alenn.mdn import MDNPosterior
from alenn.kde import KDEPosterior

#-----------------------------------------------------------------------------------------------------------------
# Main Class Source

class AdaptiveMCMC:
    '''
    A class implementing the adaptive MCMC algorithm.
    
    Attributes
    ----------
    S : int
        Number of sample sets.
    K : int
        Number of samples per set.
    posterior : MDNPosterior or KDEPosterior object
        A posterior approximator object with all required attributes (model,
        prior, and data) set.
    theta_lower : 1-d numpy array
        Lower bounds for each parameter.
    theta_upper : 1-d numpy array
        Upper bounds for each parameter.
    param_samples : list
        Raw parameter samples.
    likelihood_samples : list
        Raw log-likelihood samples.
        
    Methods
    -------
    set_posterior(self, posterior)
        Set the posterior approximator to be used during sampling.
    set_initialisation_ranges(self, theta_lower, theta_upper)
        Sets the ranges over which to generate the initial sample set.
    sample_posterior(self)
        Generate the parameter and log-likelihood posterior samples.
    process_samples(self, burn_in)
        Process the posterior samples into a user-friendly format.
    '''
    
    #-------------------------------------------------------------------------------------------------------------
    # Class Constructor

    def __init__(self, S = 5000, K = 70):
        '''
        Parameters
        ----------
        S : int
            Number of sample sets.
        K : int
            Number of samples per set. 

        Returns
        -------
        None.
        
        Calling the class constructor with no arguments simply loads the default 
        sampler configuration employed in the paper (S = 5000, K = 70).
        '''
        
        # Set Sampler Attributes
        self.S = S
        self.K = K
        
        # Display an Object Creation Message to the User
        print('-----------------------------------------------')
        print('Successfully created a new MCMC sampler object:')
        print('-----------------------------------------------')

        print('{: <30} {: <20}'.format('Number of sample sets:', S))
        print('{: <30} {: <20}'.format('Number of samples per set:', K))
        
        print('-----------------------------------------------')
        print('')
        
    #-------------------------------------------------------------------------------------------------------------
    # User-called Methods

    def set_posterior(self, posterior):
        '''
        Set the posterior approximator to be used during sampling.
        
        Parameters
        ----------
        posterior : MDNPosterior or KDEPosterior object
            A posterior approximator object with all required attributes (model,
            prior, and data) set.

        Returns
        -------
        None.
        '''
        
        # Attempt to Create a New Attribute to Store the Provided Posterior Object
        if isinstance(posterior, MDNPosterior) or isinstance(posterior, KDEPosterior):
            self.posterior = posterior
            print('{} object successfully loaded.'.format(posterior.__class__.__name__))
            print('----------------------------------------------------------------------------')
            print('')
        else:
            print('Error: Provided argument is not an MDNPosterior or KDEPosterior object.')
            
    def set_initialisation_ranges(self, theta_lower, theta_upper):
        '''
        Set the ranges over which to generate the initial sample set.
        
        Parameters
        ----------
        theta_lower : 1-d numpy array
            Lower bounds for each parameter.
        theta_upper : 1-d numpy array
            Upper bounds for each parameter.

        Returns
        -------
        None.
        '''
        
        # Attempt to Create New Attributes to Store the Provided Initialisation Ranges
        try:
            
            # Check that the Initialisation Ranges are 1-d
            if len(theta_lower.shape) == 1 and len(theta_upper.shape) == 1:
                
                # Check that the Correct Number of Bounds are Specified
                if theta_lower.shape[0] == self.posterior.num_param and theta_lower.shape[0] == self.posterior.num_param:
                    
                    # Set the Initialisation Ranges
                    self.theta_lower = theta_lower
                    self.theta_upper = theta_upper
                
                    # Display a Success Message
                    print('Initialisation ranges successfully set.')
                    print('')
                                        
                    # Create a Parameter Bound Table
                    bound_table = pd.DataFrame(np.array([theta_lower, theta_upper]).transpose(), columns = ['Lower Bound', 'Upper Bound'])
                    bound_table.index += 1
                    bound_table.index.name = 'Parameter'
                    
                    # Display Chosen Parameter Ranges
                    print(bound_table)
                    print('----------------------------------------------------------------------------')
                    print('')
                
                else:
                    
                    # Display a Failure Message
                    print('Error: The initialisation bound dimensions do not match the number of model parameters.')   
                      
            else:
                
                # Display a Failure Message
                print('Error: Provided initialisation ranges are not 1-d numpy arrays.')
         
        except:
            
            # Display a Failure Message
            print('Error: Provided initialisation ranges are not 1-d numpy arrays.')
            
    def sample_posterior(self):
        '''
        Generate the parameter and log-likelihood posterior samples and store them 
        as the attributes param_samples and likelihood_samples respectively.
        
        Returns
        -------
        None.
        '''
        
        # Generate Random Initial Guesses
        initial_guess = np.random.uniform(self.theta_lower, self.theta_upper, [self.K, self.posterior.num_param]).transpose()
    
        # Create a Data Structure to Store the Current Log-Posterior Probabilities and Log-Likelihoods of Each Sample
        p_current = []
        ll_initial = []
        
        # Set-up a Progress Bar
        with tnrange(self.K, desc = 'Initial Posterior Evaluations') as initial_evals:
            
            # Perform the Initial Log-Posterior and Log-Likelihood Evaluations
            for i in initial_evals:
                
                # Account for a Possible Simulation Crash
                try:
                    curr_eval_output = self.posterior.evaluate_posterior(initial_guess[:, i])
                    p_current.append(curr_eval_output[0])
                    ll_initial.append(curr_eval_output[1])
                    initial_evals.set_postfix(log_likelihood = str(np.round(curr_eval_output[1], 4)))
                except:
                    p_current.append(-np.inf)
                    ll_initial.append(np.nan)
                    initial_evals.set_postfix(log_likelihood = -np.inf)
                    
        # Create Data Structures to Store the Markov Chains
        theta_chain = [initial_guess]
        ll_chain = [np.array(ll_initial)]
                
        # Set-up a Progress Bar
        with tnrange(self.S, desc = 'Posterior Sampling') as main_evals:
            
            # Apply the Adaptive Metropolis-Hastings Algorithm for S Iterations
            for i in main_evals:
        
                # Construct the Proposal Distribution
                proposal = stats.gaussian_kde(theta_chain[-1], 'silverman')
        
                # Generate the Next Candidate
                theta_c = proposal.resample(1).flatten()
                
                # Determine the Candidate Log-Posterior and Log-Likelihood
                try:
                    curr_eval_output = self.posterior.evaluate_posterior(theta_c)
                    p_c = curr_eval_output[0]
                except:
                    curr_eval_output = [-np.inf, np.nan]
                    p_c = curr_eval_output[0]

                # Choose a Candidate Copy
                k = np.random.randint(self.K)
        
                # Calculate the Proposal Component of the Denominator 
                prop_denom = proposal.pdf(theta_c)
        
                # Calculate the Proposal Component of the Numerator
                prop_num_samples = theta_chain[-1].copy()
                prop_num_samples[:, k] = theta_c
                prop_num_dist = stats.gaussian_kde(prop_num_samples, 'silverman')
                prop_num = prop_num_dist.pdf(theta_chain[-1][:, k])
        
                # Determine the Acceptance Probability
                if prop_num / proposal.pdf(theta_chain[-1][:, k]) > 1e-8:
                    alpha = min([1, np.exp(p_c - p_current[k]) * np.divide(prop_num, prop_denom)])
                else:
                    alpha = 1
                
                # When determining the acceptance probability above, we also apply the correction
                # discussed in Appendix 2 and a log-sum-exp trick to avoid underflows.
                
                # Generate a New Base Sample Set
                theta_chain.append(theta_chain[-1].copy())
                ll_chain.append(ll_chain[-1].copy())
        
                # Determine if the Candidate is Accepted and Update the Sample Set
                if np.random.uniform() <= alpha:
                    theta_chain[-1][:, k] = theta_c.copy()
                    ll_chain[-1][k] = curr_eval_output[1]
                    p_current[k] = p_c
    
                # Update the Progress Bar Status Messages
                main_evals.set_postfix(log_likelihood = str(np.round(curr_eval_output[1], 4)))
         
        # Store the Generated Samples
        self.param_samples = theta_chain
        self.likelihood_samples = ll_chain
        
    def process_samples(self, burn_in):
        '''
        Process the posterior samples stored in the param_samples and likelihood_samples
        attributes into a user-friendly format.
        
        Parameters
        ----------
        burn_in : int
            Number of sample sets to discard as part of a burn-in period.

        Returns
        -------
        sample_matrix : 2-d numpy array
            A matrix consisting of the parameter and log-likelihood samples 
            obtained after discarding sample sets that constitute the burn-in 
            period. The matrix columns represent, in order, the model parameters 
            and the final column corresponds to the log-likelihood samples.
        '''
        
        # Process the Parameter Samples
        sample_matrix = [np.array([self.param_samples[i][j, :] for i in range(burn_in + 1, len(self.param_samples))]).flatten() for j in range(self.posterior.num_param)]
            
        # Process the Log-Likelihood Samples
        sample_matrix.append(np.array([self.likelihood_samples[i] for i in range(burn_in + 1, len(self.param_samples))]).flatten())
            
        # Format and Return the Sample Matrix
        return np.array(sample_matrix).transpose()