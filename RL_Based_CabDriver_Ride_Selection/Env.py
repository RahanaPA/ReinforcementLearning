# Import routines

import numpy as np
import math
import random

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [(i,j) for i in range(1,m+1) for j in range(1,m+1) if i!=j]
        self.action_space.insert(0, (0,0))
        self.state_space = [(i, j, k) for i in range(1,m+1) for j in range(0,t) for k in range(0,d)] 
        self.state_init =self.state_space[np.random.randint(len(self.state_space))] 

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        location=np.zeros(m)
        location[state[0]-1]=1

        hod=np.zeros(t)
        hod[state[1]]=1

        dow=np.zeros(d)
        dow[state[2]]=1

        state_encod = np.hstack((location,hod,dow)).reshape(1,m+t+d)

        return state_encod


    # Use this function if you are using architecture-2 
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

        
    #     return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        requests=0
        location = state[0]
        if location == 0:
            requests = np.random.poisson(2)
        if location == 1:
            requests = np.random.poisson(12)
        if location == 2:
            requests = np.random.poisson(4)
        if location == 3:
            requests = np.random.poisson(7)
        if location == 4:
            requests = np.random.poisson(8)
        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests)+ [0] # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]

        
        #actions.append([0,0])

        return possible_actions_index,actions   



    #def reward_func(self, state, action, Time_matrix):
    #    """Takes in state, action and Time-matrix and returns the reward"""
    #    return reward
    def reward_func(self, ride_time, transit_time):
        """Takes in state, action and Time-matrix and returns the reward"""
    
        reward = (R*ride_time)-(C*(ride_time+transit_time))
        return reward



    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        next_state = []
        
        # Initialize various times
        total_time   = 0
        transit_time = 0    
        cust_time    = 0    
        ride_time    = 0    
        
        
        cur_loc=state[0]
        cur_hour=state[1]
        cur_day=state[2]

        pick_loc=action[0]
        drop_loc=action[1]
        
        if action == (0,0):

            ride_time = 0
            cust_time = 1
            
            next_time, next_day = self.modify_day_time(cur_hour,cur_day,cust_time)
            next_loc = cur_loc
          
        elif (cur_loc == pick_loc):
           
            ride_time = Time_matrix[cur_loc-1][drop_loc-1][cur_hour][cur_day]
            
            cust_time = ride_time
            
            next_time, next_day = self.modify_day_time(cur_hour,cur_day,cust_time)
            
            next_loc = drop_loc
            
        else :
            
            transit_time = Time_matrix[cur_loc-1][pick_loc-1][cur_hour][cur_day]  
              
            new_time, new_day = self.modify_day_time(cur_hour, cur_day, transit_time)
           
            ride_time = Time_matrix[pick_loc-1][drop_loc-1][new_time][new_day]
            
                       
            cust_time = transit_time + ride_time
        
            next_time, next_day = self.modify_day_time(new_time,new_day,cust_time)
            next_loc  = drop_loc
               
        
        
       
        next_state = [next_loc, next_time, next_day]
        
        return next_state, cust_time, ride_time, transit_time 
        #return next_state
        
        
        
        
        
    def modify_day_time(self, time, day, timechange):
        """
        Takes in the current state and time taken for driver's journey to return
        the state post that journey.
        """
        time = int(time + timechange)

        if time < 24:
            nexttime = int(time)
            # day is unchanged
            nextday= day    
        elif time >= 24:
            nexttime = int(time - 24)
            if day == 6:
                nextday = 0
            else:
                nextday = day + 1 
           

        return nexttime, nextday
    
    def step(self, state, action, Time_matrix):
        """
        Take a trip as cabby to get rewards next step and total time spent
        """
        # Get the next state, cust_time, transit_time, ride_time
        next_state, cust_time, ride_time, transit_time  = self.next_state_func(state, action, Time_matrix)

        # Reward 
        rewards = self.reward_func(ride_time, transit_time)
        customer_total_time = cust_time + transit_time + ride_time
        
        return rewards, next_state, customer_total_time



    def reset(self):
        return self.action_space, self.state_space, self.state_init
