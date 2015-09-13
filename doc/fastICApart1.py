# this function implements the algorithm fastICA 
# fastICA runs on data using function g 
#(and dg is the derivative of g) it returns the 
# estimate of one of the initial signal
def fastICA(data,g,dg):
    data = center(data)
    data = mix.whiten(data)
    w = random_w(data.shape[1])

    # epsilon is used to know if w converges
    epsilon = 0
    nbsteps = 0

    # max nb of steps. Used to test different functons ( used because 
    # the algorithm might diverge)
    MAX = 100
    while(not converge(epsilon,1e-20) and nbsteps <= MAX):
        w_temp = newton_step(w,data,g,dg)
        w_temp /= np.linalg.norm(w_temp)
        epsilon = np.dot(w,w_temp)
        w = w_temp
        nbsteps += 1

    # if algorithm diverged , the nbsteps is negative
    if nbsteps > MAX:
        nbsteps = -1

    return np.dot(data,w),np.dot(data,np.array([w[1],-w[0]])),nbsteps

# center the data by substracting its mean
# return the data centered
def center(data):
    return data - data.mean(axis=0)

# return a random vector of the size of the number 
# of input channels
def random_w(n_input_channel):
    w = np.random.rand(n_input_channel)
    return w/np.linalg.norm(w)

# it converges when w and w_new point in the same 
# direction i.e. when the inner product is one
def converge(epsilon,lim):
    return abs(epsilon - 1) < lim

# implements the step 4 of the algorithm g 
# is a function and dg its derivative
def newton_step(w,data,g,dg):
    y = np.dot(data,w)
    u = np.zeros(data.shape[1])
    v = 0
    for i in range(data.shape[0]):
        u += data[i]*g(y[i])
        v += dg(y[i])
    return (u - v*w)/data.shape[0]
