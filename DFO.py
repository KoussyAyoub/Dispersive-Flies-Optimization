import numpy as np
import matplotlib.pyplot as plt
import random

# our seed for reproducable results
random.seed(10)


d = {
    0 : (10, 'Relu'), 
    1 : (10, 'Relu'),
    2 : (1, 'Sigmoid')} 

class DFO :
    '''
    this class implements the Dispersive Flies Optimization algorithm
    '''
    def __init__(self,num_flies, fitness_function,NN_structure, input_size, env_bounds, initial_weights = None, delta = 0.5 ,max_iter = 100) :

        self.num_flies = num_flies # the number of flies in our environement
        self.bounds = env_bounds # the bounds of the optimization environment
        self.max_iter = max_iter # max number of iterations
        self.delta = delta # disturbence threashold

        # we initialize the positions of flies and their fitness with an empty array
        self.fitness = np.empty([self.num_flies, 1])
        self.fitness_function = fitness_function
        
        # Our neural network structure :
        self.NN_structure = NN_structure

        # the input layer shape
        self.num_features = input_size
    
        # initialize the NN weights for each fly
        self.positions = self.initilize_flies() if initial_weights == None else initial_weights

        # initiate the best weight , and its fitness value: 
        self.best_weight = None
        self.best_fitness_value = 1000000 # big initial fitness (the bigger the worse cuz its loss of our neural network) 


    def initilize_flies(self):
        '''
        this function is used to initialize the positions of the flies following a bounded uniform distribution 
        '''
        #we inialize the weights of each fly :
        fly_weights = []
        for fly in range(self.num_flies):
            vect_size = self.num_features
            fly_weight = []
            for weight_mtx in range(len(self.NN_structure)) : 
                w = np.random.uniform(low = -self.bounds, high=self.bounds, size = (vect_size,self.NN_structure[weight_mtx][0])).tolist()
                vect_size = self.NN_structure[weight_mtx][0]
                fly_weight.append(w)
            fly_weights.append(fly_weight)
        return fly_weights

        
    def train(self, Xt, Yt):
        """
        find the best firefly position to optimize the problem. 
        """

        best_fly = 0
        for itr in range(self.max_iter):

            # we calculate the fitness of each fly 
            
            for fly in range(self.num_flies):
                self.fitness[fly] = self.fitness_function(Xt,Yt,self.positions[fly])
            

            # we extract the position of the best fly with the lowest fitness
            best_idx = np.argmin(self.fitness)

            #print("***************** the index of the best fly is ", best_idx,"*****************" )

            best_fly = self.positions[best_idx] # best weight with the lowest loss value 
            best_fly_fitness = self.fitness[best_idx] # the fitness of our best fly  

            if best_fly_fitness <= self.best_fitness_value : 
                # we update our best set of weights if the loss value is lower
                self.best_weight = best_fly
                self.best_fitness_value = best_fly_fitness


            randomized_flies = 0

            for fly in range(self.num_flies): 
                
                if fly != best_idx : 
                    left_idx, right_idx = self.positions[(fly-1)%self.num_flies], self.positions[(fly+1)%self.num_flies]  
                    Xi = left_idx if self.fitness_function(Xt, Yt, left_idx) < self.fitness_function(Xt, Yt, right_idx) else right_idx    

                    test = np.random.uniform() 

                    if test < self.delta:   # the disturbance is used for exploring new potential solutions, we select te flies to be randomized via delta
                        randomized_flies += 1 
                        for weight_idx in range(len(best_fly)) :     
                            weight_shape = self.positions[fly][weight_idx]
                            self.positions[fly][weight_idx] = np.random.uniform(-self.bounds, self.bounds, size = (len(weight_shape), len(weight_shape[0])))

            
                    else :
                        for weight_idx in range(len(best_fly)) : 
                            weight = self.positions[fly][weight_idx]
                            for row in range(len(weight)):
                                for col in range(len(weight[0])): 
                                    u = np.random.uniform(low = -self.bounds, high=self.bounds)
                                    self.positions[fly][weight_idx][row][col] = Xi[weight_idx][row][col] + u * (best_fly[weight_idx][row][col] - self.positions[fly][weight_idx][row][col] )
                                
                                    if self.positions[fly][weight_idx][row][col] < (- self.bounds) or self.positions[fly][weight_idx][row][col] > self.bounds : 
                                        self.positions[fly][weight_idx][row][col] = np.random.uniform(-self.bounds, self.bounds)
                                
            print('the fitness of the best fly in iteration ', itr ,' is  :', self.fitness_function(Xt, Yt,best_fly))
        
            print("the number of randomized flies :", randomized_flies, "out of ", self.num_flies)

        return self.best_weight, self.best_fitness_value, self.positions

if __name__ == '__main__':

    print('hey :)')





