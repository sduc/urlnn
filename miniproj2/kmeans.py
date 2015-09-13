##########################
# Title: kmeans.py       #
# Author: Sebastien Duc  #
##########################

import numpy as np
import matplotlib.pyplot as pl
from numpy.linalg import norm
from scipy import misc
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

# useful constants
DATA = 'data.txt'
BLINDDATA = 'blinddata.txt'
DATASET_SIZE = 5000
DATA_DIM = 28

# load dataset into 2-dim numpy array. Each data el is a vector
# return this array plus the array on label where label[i] is the
# label of dataset[i]
def load_data (blind = False):
    # some properties of the data

    data = DATA
    if blind:
        data = BLINDDATA

    dataset = np.loadtxt(data)
    label = np.loadtxt('labels.txt')

    if not blind:
        return dataset,label
    else:
        return dataset

# save one data element of the dataset. Store it as a 2-dim array of 28x28
def save_data_el(data_el,filename):
    # put the data_el in the right shape
    dim = np.sqrt(len(data_el))
    a = np.reshape(data_el,(dim,dim))
    # save it into a txt file
    np.savetxt(filename+'.txt',a,fmt='%3d')
    # save it into pgm file
    misc.imsave(filename+'.png',a)

# print a data element of dataset (i.e a digit)
def print_data(data_el):
    print(np.reshape(data_el,(DATA_DIM,DATA_DIM)))

# compute the shannon entropy of distribution distr
def entropy (distr):
    H = 0
    for i in distr:
        if i != 0:
            H += - i * np.log2(i)
    return H

class Clusters:

    # This class represent the clusters. They will be used to perform k-mean
    # clustering algorithm. It provides function to update prototypes and
    # clusters.
    #   ATTRIBUTES
    # dataset :         the data elements that we want to cluster
    # clusters :        array saying to which cluster belong each elements of
    #                   dataest. It contains the index of the prototype
    # prototypes :      the prototypes of the clusters
    # stats :           for each cluster, how much elements that are labeled as being
    #                   digit i
    # nb_of_changes :   the number of changes that occured during an iteration.
    #                   The ith element of the list contains the nb of changes during 
    #                   the ith iteration
    # nu :              the learing rate used by the algrithm
    # the number of changes is defined as the number of data elements that changes
    # cluster in an iteration
    def __init__( self , dataset ,prototypes ,label):
        self.dataset = dataset
        self.clusters = np.zeros(len(dataset))
        self.prototypes = np.copy(prototypes)
        self.old_prototypes = np.zeros(self.prototypes.shape)
        if label != None:
            self.stats = np.zeros((len(prototypes),np.max(label)+1))
        self.nb_of_changes = []
        self.nu = 0.2
        self.label = label
        self.iterations = 0
        self.threshold_dead_unit = 50
        self.dead_units = []

    # let data be in dataset, then this function returns the cluster for which
    # the distance is minimized
    def find_closest_cluster ( self , data_index ):
        minimum = norm(self.prototypes[0] - self.dataset[data_index])+1
        min_index = -1
        for i in range(len(self.prototypes)):
            temp = norm(self.prototypes[i] - self.dataset[data_index])
            if temp < minimum:
                minimum = temp
                min_index = i
        assert min_index != -1
        return min_index



    # assigne to each element in the dataset the nearest cluster i.e. the nearest
    # prototype.
    def update_clusters ( self, improved=False , max_it = 50):
        nchanges = 0
        self.iterations += 1
        for i in range(len(self.clusters)):
            temp = self.clusters[i]
            rand = 1
            if improved:
                rand = np.random.randint(10*len(self.clusters))
            if rand == 0 and self.iterations <= max_it:
                rand_cluster = np.random.randint(len(self.prototypes))
                self.clusters[i] = rand_cluster
            else:
                self.clusters[i] = self.find_closest_cluster(i)
            if temp != self.clusters[i]:
                nchanges += 1
        self.nb_of_changes.append(nchanges)
        self.dead_units.append(self.number_dead_units())

    # prototypes are updated and set as the center of their cluster
    def update_prototypes ( self ):
        self.old_prototypes = np.copy(self.prototypes)
        self.prototypes = np.zeros(self.prototypes.shape)
        nb_of_cluster_el = np.zeros(len(self.prototypes))
        for i in range(len(self.clusters)):
            nb_of_cluster_el[self.clusters[i]] += 1
            self.prototypes[self.clusters[i]] += self.dataset[i]
        for i in range(len(self.prototypes)):
            self.prototypes[i] /= nb_of_cluster_el[i]

    # update prototypes by using a learning rate nu which decrease at each new
    # iteration
    def learning_update_prototypes ( self , improved = False):
        self.old_prototypes = np.copy(self.prototypes)
        delta_w = np.zeros(self.prototypes.shape)
        for i in range(len(self.clusters)):
            delta_w[self.clusters[i]] = self.prototypes[self.clusters[i]] - self.dataset[i]
        self.prototypes = self.prototypes - self.nu * delta_w
        if improved:
            self.nu -= self.nu/10
            

    # use the entropy of the cluster. High entropy implies dead unit
    def entropy_clusters ( self ):
        for i in range(len(self.prototypes)):
            prob = self.stats[i]/sum(self.stats[i])
            print("For cluster "+str(i)+" the entropy is "+str(entropy(prob)))

    # clusters with a very low number of elements (i.e. the number is under some
    # threshold) are considered dead.
    def is_dead_unit ( self, cluster_i ):
        return list(self.clusters).count(cluster_i) < self.threshold_dead_unit
        
    # returns the number of dead units in the clustering  
    def number_dead_units ( self ):
        return [self.is_dead_unit(i) for i in range(len(self.prototypes))].count(True)

    # detect prototypes that represent the same digit. To do this I use a
    # correlation matrix. If two prototypes are too much correlated then they
    # have a high probability to represent the same digit
    def detect_same_class_prototypes( self ):
        # use correlation between prototypes
        return np.corrcoef(self.prototypes)


    # return true iff the algorithm has converged.
    # We say that it has converged if the number of data elements that have
    # changed cluster is 0
    def converge( self ):
        return self.nb_of_changes != [] and self.nb_of_changes[-1] < 1 #(np.sort(self.old_prototypes) == np.sort(self.prototypes)).all()

    # given a data element with a data index (i.e. the ith element in the
    # database) return its associated prototype
    def get_prototype ( self , data_index ):
        return self.prototypes[cluster[data_index]]

    def __str__ ( self ):
        out = "The algorithm converged with the following clusters \n"
        for i in range(len(self.prototypes)):
            out += "  Cluster " + str(i) + " : \n"
            for j in range(len(self.dataset)):
                if self.clusters[j] == i:
                    #out += "    data index : "+str(j)+" ,data label :"+str(label[j])+"\n"
                    self.stats[i,self.label[j]] += 1 
            for j in range(self.stats.shape[1]):
                out += "    Digit "+str(j)+" appears "+str(self.stats[i,j])+" times\n"
            out+="\n"
        return out


    def __repr__ ( self ):
        return str(self)


# k-means clustering algorithm
def kmeans(k,dataset,label,improved=False):
    proto = initialize_prototypes(k,dataset,label, False)
    # for each data element, what is the index of its associated prototype (in
    # proto)
    clusters = Clusters(dataset , proto , label)
    i = 1
    while not clusters.converge():
        print("Iterated " + str(i) + " times")
        clusters.update_clusters(improved)
        print("- after update clusters")
        clusters.learning_update_prototypes(improved)
        #clusters.update_prototypes()
        print("- after update prototypes")
        i += 1
    return clusters
    
# return the initial array of the prototypes. Initially they are set to k random
# data elements of dataset
def initialize_prototypes(k,dataset,label,improved = False):
    
    def min_norm (prototypes,x):
        return min([int(norm(dataset[p] - x)**2) for p in prototypes])

    print("Initialization of the prototypes")
    prototypes = np.zeros(k,dtype=int)
    if improved:
        init = np.random.randint(len(dataset))
        prototypes[0] = init
        i = 1
        while i < k:
            print(i)
            distances = [min_norm(prototypes[:i],x) for x in dataset]
            rand = np.random.randint(sum(distances))
            j = 0
            found = False
            while not found:
                # we don't consider prototypes already chosen
                while j in prototypes[:i]:
                    j += 1
                if rand <= sum(distances[:j+1]):
                    found = True
                else:
                    j += 1
            prototypes[i] = j
            i += 1
    else:
        prototypes = np.random.randint(0,len(dataset),k)
    #print(prototypes)
    #print(label[prototypes])
    return dataset[prototypes] 

# save the learned prototypes in different files
def save_prototypes(clusters):
    if 'output' not in os.listdir('.'):
        os.mkdir('output')
    for i in range(len(clusters.prototypes)):
        save_data_el(clusters.prototypes[i],'output/prototype'+ str(i))


############################################################################
#                 PCA
############################
# this function performs pca on X and keeps only dim dimensions
def pca ( X, dim ):
    assert dim < X.shape[1]
    Y = X - np.mean(X,axis = 0)
    C = np.cov(X.T)
    n,v = np.linalg.eigh(C)
    Y = Y.dot(v).T
    return Y[-dim:].T

# reduce the size and the dimension of dataset. We keep only digit 0, 1, 2, 3.
# Then we apply PCA and we keep only the dim prinicpal componants
def reduce_dim ( dataset , label , dim ):
    size = len(filter(lambda x: x==0 or x==1 or x==2 or x ==3, list(label)))
    reduced_dataset = np.zeros((size,dataset.shape[1]))
    reduced_label = np.zeros(size)
    index = 0
    for i in range(len(dataset)):   
        if label[i] == 0 or label[i] == 1 or label[i] == 2 or label[i] == 3:
            reduced_dataset[index] = np.copy(dataset[i])
            reduced_label[index] = label[i]
            index += 1
    assert index == size

    # apply PCA 
    return pca( reduced_dataset,dim), reduced_label


def plot_res (clusters,dim,n_rows,n_cols):
    # plot the number of changes per step (we don't consider the first)
    pl.figure(1)
    pl.bar(np.arange(1,len(clusters.nb_of_changes)),clusters.nb_of_changes[1:])
    
    # plot the prototypes
    pl.figure(2)
    for i in range(len(clusters.prototypes)):
        pl.subplot(n_rows,n_cols,i+1)
        pl.imshow(np.reshape(clusters.prototypes[i],(dim,dim)),cmap=pl.cm.gray)
        pl.axis('off')
    pl.figure(3)
    pl.bar(np.arange(len(clusters.dead_units)),clusters.dead_units)
    pl.show()



############################################################################
#                                MAIN                                      #
############################################################################

def normal(improved):
    dataset,label = load_data()

    clusters = kmeans(20,dataset,label,improved)

    save_prototypes(clusters)
    print(clusters)
    clusters.entropy_clusters()
    print(clusters.detect_same_class_prototypes())
    print("number of dead units : " + str(clusters.number_dead_units()))

    plot_res(clusters,28,4,5)

def reduced(improved):
    dataset,label = load_data()

    print("--Reduce the data set--")
    red_dset,reduced_label = reduce_dim(dataset,label,64)

    #3d plot
    subset = [0,1,3]
    colors = ['.r','.b','.g','.y']
    fig = figure()
    ax = Axes3D(fig)
    #plot each digit with a different color
    for i in range(4):
        size = len(filter(lambda x: x==i, list(reduced_label)))
        label2plot = np.zeros((size,red_dset.shape[1]))
        index = 0
        for j in range(len(red_dset)):
            if reduced_label[j] == i:
                label2plot[index] = np.copy(red_dset[j])
                index += 1
        ax.plot(label2plot.T[subset[0]],label2plot.T[subset[1]],label2plot.T[subset[2]],colors[i])
    pl.show()

    print("--Apply k-means on the reduced set--")
    clusters = kmeans(10,red_dset,reduced_label,improved)

    print(clusters)
    clusters.entropy_clusters()
    #plot the result
    plot_res(clusters,8,2,5)

# If you want to use reduction then reduction = dimension where dimension is the
# number of dimensions you want to keep
def blinddata(reduction=None):
    dataset = load_data(blind = True)
    if reduction != None:
        dataset = pca(dataset,reduction)

    clusters = kmeans(20,dataset,None,False)
    print("number of dead units : "+str(clusters.number_dead_units()))
    for i in range(len(clusters.prototypes)):
        if clusters.is_dead_unit(i):
            print("- cluster "+str(i)+" is probably a dead unit")
    
    #save that prototype matrix in a file
    np.savetxt("blindprototypes.txt",clusters.prototypes,fmt="%3d")


###########################################
# uncomment the function you want to use ##
###########################################

reduced(False)
#normal(True)
#blinddata()
