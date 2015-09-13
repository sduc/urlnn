# this function implements the algorithm fastICA 
# for all components it returns the unmixed signals
def fastICAall(data,g,dg):
    nIC = data.shape[1]

    data = center(data)
    data = mix.whiten(data)
    w = np.array([random_w(nIC) for i in range(nIC)])

    epsilon = 0
    w_temp = np.zeros_like(w)
    while(not converge(epsilon)):
        print("Epsilon is currently ", epsilon)
        for i in range(nIC):
            w_temp[i] = newton_step(w[i],data,g,dg)
            w_temp[i] /= np.linalg.norm(w_temp[i])
        W = sym_decorrelation(w_temp)
        epsilon = max(abs(np.diag(np.dot(W, w.T))))
        w = np.copy(W)

    return np.array([np.dot(data,w[i]) 
                     for i in range(nIC)])
        
# symmetric decorrelation
def sym_decorrelation(W):
    K = np.dot(W, W.T)
    s, u = np.linalg.eigh(K)
    # u (resp. s) contains the eigenvectors (resp. 
    # square roots of the eigenvalues) 
    u, W = [np.asmatrix(e) for e in (u, W)]
    W = (u * np.diag(1.0/np.sqrt(s)) * u.T) * W 
    return np.asarray(W)
