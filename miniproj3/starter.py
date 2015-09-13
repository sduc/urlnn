import sys
import pylab as plb
import numpy as np
import mountaincar
from math import exp

class DummyAgent():
    """A not so good agent for the mountain-car task.
    """

    def __init__(self, mountain_car = None, parameter1 = 3.0\
                 , N_input_neurons = 10, eta=0.1, lambda_e = 0.95\
                 , tau = 8, w = 1):
        
        if mountain_car is None:
            self.mountain_car = mountaincar.MountainCar()
        else:
            self.mountain_car = mountain_car

        self.parameter1 = parameter1
        
        self.N_input_neurons = N_input_neurons
        self.eta = eta
        self.lambda_e = lambda_e
        self.gamma = 0.95
        self.tau = tau
        
        self.latencies = []

        self.input_layer = InputLayer(N_input_neurons,init_w=w)
        self.action = None

        self.stop = False

    def reinitialize_agent(self,w):
        self.mountain_car = mountaincar.MountainCar()
        self.input_layer = InputLayer(self.N_input_neurons,init_w=w)
        self.latencies = []
        self.action = None

    def visualize_trial(self, n_steps = 200):
        """Do a trial without learning, with display.

        Parameters
        ----------
        n_steps -- number of steps to simulate for
        """
        
        # prepare for the visualization
        plb.ion()
        mv = mountaincar.MountainCarViewer(self.mountain_car)
        mv.create_figure(n_steps, n_steps)
        plb.draw()
            
        # make sure the mountain-car is reset
        self.mountain_car.reset()

        for n in range(n_steps):
            print '\rt =', self.mountain_car.t,
            sys.stdout.flush()
            
            # choose a random action
            self.mountain_car.apply_force(np.random.randint(3) - 1)
            # simulate the timestep
            self.mountain_car.simulate_timesteps(100, 0.01)

            # update the visualization
            mv.update_figure()
            plb.draw()            
            
            # check for rewards
            if self.mountain_car.R > 0.0:
                print "\rreward obtained at t = ", self.mountain_car.t
                break

    def learn(self, N_trials = 20):
        for t in range(N_trials):
            latency = self.run_trial()
            self.latencies.append(latency)
            # decay the exploration temperature
            if self.tau > 2**(-6):
                self.tau *= 0.5

    # run one trial and return the latency
    def run_trial(self):
        self.initialize_trial()
        self.choose_action()
        while not self.arrived() and not self.stop:
            print '\rt =', self.mountain_car.t,
            sys.stdout.flush()

            self.update_state()
            self.choose_action()
            self.update_weights()
        print "\rreward obtained at t = ", self.mountain_car.t
        return self.mountain_car.t

    def initialize_trial(self):
        self.mountain_car.reset()
        self.stop = False

    def arrived(self):
        return self.mountain_car.R > 0.0

    def update_state(self):
        self.old_x = self.mountain_car.x
        self.old_x_d = self.mountain_car.x_d
        self.mountain_car.apply_force(self.action)
        self.mountain_car.simulate_timesteps(100,0.01)

    def choose_action(self):
        self.action_old = self.action
        s = (self.mountain_car.x,self.mountain_car.x_d)
        p = [np.exp(self.input_layer.getQ(s,j)/self.tau) for j in range(-1,2)]
        distr = [sum(p[:i+1]) for i in range(len(p))]
        p_a = np.random.rand()*distr[-1]
        self.action = None
        for i in range(len(distr)):
            if p_a <= distr[i]:
                self.action = i
                break
        assert self.action is not None
        self.action -= 1

    # if the agent gets stuck then give a negative reward and stop the process
    # return the reward.
    def stop_agent(self):
        if self.mountain_car.t > 5000:
            self.stop = True
            return -1
        else:
            return self.mountain_car.R

    def update_weights(self):
        assert self.action_old != None
        #update eligibility trace
        self.input_layer.update_e(self.gamma*self.lambda_e)
        old_s = (self.old_x,self.old_x_d)
        r_post = self.input_layer.getQ(old_s,self.action_old)
        for j in range(self.N_input_neurons**2):
            r_pre = self.input_layer.get_activity(j,old_s)
            self.input_layer.set_e(self.action_old,j,r_pre)#*r_post)

        #update weights
        #print self.input_layer.weights
        R = self.stop_agent()
        s = (self.mountain_car.x,self.mountain_car.x_d)
        self.input_layer.weights += self.eta * self.input_layer.e *\
                (R-\
                (r_post - self.gamma *self.input_layer.getQ(s,self.action)))


# this class reprensent the neurons of the input layer
class InputLayer():

    def __init__(self, N, init_w=0 ):
        self.weights = None
        self.N = N
        self.initialize_weights(init_w)
        self.positions = np.zeros((N**2,2))
        self.sigma_x = None
        self.sigma_xi = None
        self.initialize_positions()
        self.e = np.zeros((3,N**2))
        """ PLOT MAP OF NEURONS
        ax = plb.subplot(111)
        ax.scatter(self.positions[:,0],self.positions[:,1])
        plb.grid(True)
        plb.show()
        """

    def initialize_weights(self, w):
        if w == 0:
            self.weights = np.zeros((3,self.N**2))
        else:
            self.weights = np.ones((3,self.N**2))

    # initialize the positions of the neurons on the grid. On axis x, the
    # centers must be between -150 and 30. On axis x_dot, the centers must be
    # between -15 ad 15.
    def initialize_positions(self):
        self.sigma_x = ( 30 - (-150) )*1.0/( self.N )
        self.sigma_xi = ( 15 - (-15) )*1.0/( self.N )
        for i in range(self.N):
            for j in range(self.N):
                self.positions[(self.N*i) + j,0] = -150 + (i+0.5)*self.sigma_x
                self.positions[(self.N*i) + j,1] = -15 + (j+0.5)*self.sigma_xi

    # get the activity of the neuron j, when the car is in state s=(x,x_dot) 
    def get_activity(self,j,s):
        x = s[0]
        x_dot = s[1]
        r = exp(-((self.get_position_x(j)-x)/self.sigma_x)**2 -\
                   ((self.get_position_x_dot(j)-x_dot)/self.sigma_xi)**2)
        assert r<=1 and r>=0
        return r

    # get the weight of neuron j when taking action a \in {-1,0,1}
    def get_weight(self,a,j):
        return self.weights[a+1,j]

    # set the weight of neuron j when taking action a to new_value
    def set_weight(self,a,j,new_value):
        self.weights[a+1,j] = new_value

    def get_position_x(self,j):
        return self.positions[j,0]

    def get_position_x_dot(self,j):
        return self.positions[j,1]

    def getQ(self,s,a):
        return sum([self.get_weight(a,j)*self.get_activity(j,s)\
                    for j in range(self.N**2)])

    def init_e(self):
        self.e = np.zeros((3,self.N**2))

    def get_e(self,a,j):
        return self.e[a+1,j]

    def set_e(self,a,j,new_value):
        self.e[a+1,j] += new_value

    def update_e(self,lambda_e):
        self.e = lambda_e*self.e


if __name__ == "__main__":
    d = DummyAgent(lambda_e=0.95)
    #d.learn()
    #d.visualize_trial()

    print("-- Simulation of agent n 1 --")
    Nagent = 10
    d.learn()
    latencies = np.array(d.latencies)
    for i in range(Nagent-1):
        print "-- Simulation of agent n ",i+2," --"
        d.reinitialize_agent(1)
        d.learn()
        latencies += np.array(d.latencies)
    latencies /= Nagent
    plb.bar(np.arange(len(latencies)),latencies)
    plb.show()
