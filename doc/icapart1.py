def square_cos_ica(g,dg):
    data = mix.mixsquarecos()
    x = f.fastICA(data,g,dg)
    return x

# g and its derivative used for the algorithm
g1 = lambda x: m.tanh(x) 
dg1 = lambda x: 1 - m.tanh(x)**2 

g2 = lambda x: x*m.exp(-(x**2)/2)
dg2 = lambda x: -m.exp(-(x**2)/2)*(x**2 - 1) 

g3 = lambda y: y**3
dg3 = lambda y: 3*y**2


n = 200 # samples to plot
def plot(sig):
    pl.plot(np.arange(n), sig[:n])
    pl.show()

# run the algorithm and plot the result
sig1 = square_cos_ica(g1,dg1)
plot(sig1)
