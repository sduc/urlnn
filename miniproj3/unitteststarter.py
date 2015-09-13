import unittest
import numpy as np
from starter import *
from numpy.random import *
import math as m
import pylab as pl

class InputLayerTestCase(unittest.TestCase):
    def setUp(self):
        self.inputlayer = InputLayer(20)

    def tearDown(self):
        self.inputlayer = None

    def test_activity(self):
        for s in zip(rand(1000)*(30+150)-150,rand(1000)*(15+15)-15):
            activities =np.array([self.inputlayer.get_activity(j,s) for j in\
                         range(self.inputlayer.N**2)])
            self.assertTrue(np.all(activities<=1) and\
                    np.all(activities >=0))
            self.assertTrue(np.any(activities>=m.exp(-0.5)))
            self.assertTrue(np.any(activities<=m.exp(-0.5*(self.inputlayer.N-1)**2)))

    def test_update_e(self):
        old_e = np.copy(self.inputlayer.e)
        cste = rand()
        self.inputlayer.update_e(cste)
        for i in range(old_e.shape[0]):
            for j in range(old_e.shape[1]):
                self.assertTrue(old_e[i,j]*cste ==self.inputlayer.get_e(i-1,j))


class DummyAgentTestCase(unittest.TestCase):
    def setUp(self):
        self.agent = DummyAgent()

    def tearDown(self):
        self.agent = None

    def test_choose_action(self):
        #run 100 steps of the algorithm
        self.agent.initialize_trial()
        self.agent.choose_action()
        i = 0
        while i < 100:
            self.agent.update_state()
            self.agent.choose_action()
            self.agent.update_weights()
            i+=1

        self.agent.update_state()
        stats = np.zeros(3)
        for i in range(2000):
            self.agent.choose_action()
            stats[self.agent.action+1] += 1
        s = (self.agent.mountain_car.x,self.agent.mountain_car.x_d)
        p = [np.exp(self.agent.input_layer.getQ(s,j)/self.agent.tau) for j in range(-1,2)]
        pl.figure(1)
        pl.bar(np.arange(len(p)),p)
        pl.figure(2)
        pl.bar(np.arange(len(stats)),stats)
        pl.show()
        



        
suite1 = unittest.TestLoader().loadTestsFromTestCase(InputLayerTestCase)
suite2 = unittest.TestLoader().loadTestsFromTestCase(DummyAgentTestCase)
unittest.TextTestRunner(verbosity=2).run(suite2)

