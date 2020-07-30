import GPy
import GPyOpt
import numpy as np
import time
from numpy.random import seed
import matplotlib
from bayesmark.abstract_optimizer import AbstractOptimizer
from bayesmark.experiment import experiment_main
from bayesmark.space import JointSpace
from GPyOpt.acquisitions.EI import AcquisitionEI
from copy import deepcopy
import math


class BoEI(AbstractOptimizer):
    primary_import = None
    
    def __init__(self, api_config):
        """Build wrapper class to use optimizer in benchmark.

        Parameters
        ----------
        api_config : dict-like of dict-like
            Configuration of the optimization variables. See API description.
        """
        AbstractOptimizer.__init__(self, api_config)
        
#         print ('api_config: ', api_config)
        api_space = BoEI.api_manipulator(api_config)  # used for GPyOpt initialization
#         print('api_space: ', api_space)
        self.space_x = JointSpace(api_config) # used for warping & unwarping of new suggestions & observations
#         print('space_x: ', self.space_x)
        self.hasCat, self.cat_vec = BoEI.is_cat(api_config)
#         print('cat_vec: ', self.cat_vec)
        
        self.dim = len(self.space_x.get_bounds())
        
#         self.func = GPyOpt.objective_examples.experiments2d.branin()
        
#         self.objective = GPyOpt.core.task.SingleObjective(self.func.f)
        self.objective = GPyOpt.core.task.SingleObjective(None)

        self.space = GPyOpt.Design_space(api_space)
        
        self.model = GPyOpt.models.GPModel(optimize_restarts=5,verbose=False)
        
        self.aquisition_optimizer = GPyOpt.optimization.AcquisitionOptimizer(self.space)
        
        
        self.aquisition = AcquisitionEI(self.model, self.space, optimizer=self.aquisition_optimizer, cost_withGradients=None)
        
        self.batch_size = None
        
#         self.max_iter = 16
        
#         self.max_time = np.inf

        
    # return an array that indicates which variable is cat/ordinal
    @staticmethod
    def is_cat(API):
        api_items = API.items()
        cat_vec = []
        counter = 0
        hasCat = False

        for item in api_items:
            if item[1]['type'] == 'cat':
                hasCat = True
                singleCat = [counter, len(item[1]['values'])]
                cat_vec.append(singleCat)
            counter += 1            
        cat_vec = np.array(cat_vec)
        
        return hasCat, cat_vec

        
    @staticmethod
    def api_manipulator(api_config):
        api = deepcopy(api_config)
        api_space = []
        api_items = api.items()
        api_items = sorted(api_items)   # make sure the entries are aligned with warpping
        for item in api_items:
            variable = {}
    
            # get name
            variable['name'] = item[0]
            if item[1]['type'] == 'real':                  #real input
                if 'range' in item[1]:
                    variable['type'] = 'continuous'       #continuous domain
                if 'values' in item[1]:
                    variable['type'] = 'discrete'         #discrete domain         
            elif item[1]['type'] == 'int':                # int input
                variable['type'] = 'discrete'
                if 'range' in item[1]:                    #tranform into discrete domain
                    lb = item[1]['range'][0]
                    ub = item[1]['range'][1]
                    values_array = np.arange(lb, ub+1)
                    del item[1]['range']
                    item[1]['values'] = values_array            
            elif item[1]['type'] == 'cat':
                variable['type'] = 'categorical'        
            elif item[1]['type'] == 'bool':
                variable['type'] = 'categorical'
         
            #transform space
            if (item[1]['type'] == 'real') or (item[1]['type'] == 'int'):
                if item[1]['space'] == 'log':
                    if 'range' in item[1]:
                        lb = item[1]['range'][0]
                        ub = item[1]['range'][1]
                        assert lb > 0
                        assert ub > 0
                        item[1]['range'] = (math.log(lb), math.log(ub))
                    if 'values' in item[1]:
                        item[1]['values'] = np.log(item[1]['values'])
                if item[1]['space'] == 'logit':
                    if 'range' in item[1]:
                        lb = item[1]['range'][0]
                        ub = item[1]['range'][1] 
                        assert lb > 0 and lb < 1
                        assert ub > 0 and lb < 1
                        lb_new = math.log(lb/(1.0-lb))
                        ub_new = math.log(ub/(1.0-ub))
                        item[1]['range'] = (lb_new, ub_new)
                    if 'values' in item[1]:
                        values_arr = item[1]['values']
                        item[1]['values'] = np.log(values_arr/(1.0-values_arr))
                if item[1]['space'] == 'bilog':
                    if 'range' in item[1]:
                        lb = item[1]['range'][0]
                        ub = item[1]['range'][1] 
                        lb_new = math.log(1.0+lb) if lb >= 0.0 else -math.log(1.0-lb)
                        ub_new = math.log(1.0+ub) if ub >= 0.0 else -math.log(1.0-ub)
                        item[1]['range'] = (lb_new, ub_new)
                    if 'values' in item[1]:
                        values_arr = item[1]['values']
                        item[1]['values'] = np.sign(values_arr) * np.log(1.0 + np.abs(values_arr))
                        
   
            #get domain
            if (item[1]['type'] == 'real') or (item[1]['type'] == 'int'):        
                if 'range' in item[1]:
                    variable['domain'] = item[1]['range']
                if 'values' in item[1]:
                    variable['domain'] = tuple(item[1]['values'])
            
            if item[1]['type'] == 'cat':
                ub = len(item[1]['values'])
                values_array = np.arange(0, ub)
                variable['domain'] = tuple(values_array)
        
            if item[1]['type'] == 'bool':
                variable['domain'] = (0, 1)
    
            api_space.append(variable)
        
        
        return api_space
        
        
    def suggest(self, n_suggestions=1):
        """Get suggestions from the optimizer.

        Parameters
        ----------
        n_suggestions : int
            Desired number of parallel suggestions in the output

        Returns
        -------
        next_guess : list of dict
            List of `n_suggestions` suggestions to evaluate the objective
            function. Each suggestion is a dictionary where each key
            corresponds to a parameter being optimized.
        """
       
        if self.batch_size is None:
#             print('first suggest')
            next_guess = GPyOpt.experiment_design.initial_design('random', self.space, n_suggestions)
        else:
#             print('other suggest')
            next_guess = self.bo._compute_next_evaluations()#in the shape of np.zeros((n_suggestions, self.dim))
    



        # preprocess the array from GpyOpt for unwarpping
        if self.hasCat == True:
            new_suggest = []
            cat_vec_pos = self.cat_vec[:,0]
            cat_vec_len = self.cat_vec[:,1]
            
            # for each suggstion in the batch
            for i in range(len(next_guess)):
                index = 0
                single_suggest = []
                # parsing through suggestions to replace the cat ones to the usable format 
                for j in range(len(next_guess[0])):
                    if j != cat_vec_pos[index]:
                        single_suggest.append(next_guess[0][j])
                    else:
                        # if a cat varible
                        value = next_guess[i][j]
                        vec = [0.]*cat_vec_len[index]
                        vec[value] = 1.
                        single_suggest.extend(vec)
                        index += 1
                        index = min(index, len(cat_vec_pos)-1)
                # asserting the desired length of the suggestion
                assert len(single_suggest) == len(next_guess[0])+sum(cat_vec_len)-len(self.cat_vec)
                new_suggest.append(single_suggest)
            assert len(new_suggest) == len(next_guess)
           
            new_suggest = np.array(new_suggest).reshape(len(suggest), len(suggest[0])+sum(cat_vec_len)-len(self.cat_vec))
            next_guess = new_suggest
             

        suggestions = self.space_x.unwarp(next_guess)
#         print("suggest: ", suggestions)
        
        return suggestions
    
    
    def observe(self, X, y):
        """Feed an observation back.

        Parameters
        ----------
        X : list of dict-like
            Places where the objective function has already been evaluated.
            Each suggestion is a dictionary where each key corresponds to a
            parameter being optimized.
        y : array-like, shape (n,)
            Corresponding values where objective has been evaluated
        """
        assert len(X) == len(y)

#         print('X', X)
#         print('y', y)
        
        XX = self.space_x.warp(X)
        yy = np.array(y)[:, None]


        # preprocess XX after warpping for GpyOpt to use
        if self.hasCat == True:
            new_XX = np.zeros((len(XX), self.dim))
            cat_vec_pos = self.cat_vec[:,0]
            cat_vec_len = self.cat_vec[:,1]

            for i in range(len(XX)):
                index = 0  # for cat_vec
                traverse = 0 # for XX
                for j in range(self.dim):
                    if j != cat_vec_pos[index]:
                        new_XX[i][j] = int(XX[i][traverse])
                        traverse += 1
                    else:
                        for v in range(cat_vec_len[index]):
                            if XX[i][traverse + v] == 1.0:
                                new_XX[i][j] = int(v)
                        traverse += cat_vec_len[index]
                        index += 1
                        index = min(index, len(cat_vec_pos)-1)

            XX = new_XX 
            
        if self.batch_size is None:
            self.X_init = XX
            self.batch_size = len(XX)
            self.Y_init = yy
            # evaluator useless but need for GPyOpt instantiation
            self.evaluator = GPyOpt.core.evaluators.RandomBatch(acquisition=self.aquisition, batch_size = self.batch_size)
            self.bo = GPyOpt.methods.ModularBayesianOptimization(self.model, self.space, self.objective, self.aquisition, self.evaluator, self.X_init, Y_init=self.Y_init)
            self.X = self.X_init
            self.Y = self.Y_init
        else:
            # update the stack of all the evaluated X's and y's
            self.bo.X = np.vstack((self.bo.X, deepcopy(XX)))
            self.bo.Y = np.vstack((self.bo.Y, deepcopy(yy)))
            # update GP model
            
      
        
        # update GP model
        self.bo._update_model('stats')
        # bo has attribute bo.num_acquisitions        
        self.bo.num_acquisitions += 1  
        
if __name__ == "__main__":
    experiment_main(BoEI)