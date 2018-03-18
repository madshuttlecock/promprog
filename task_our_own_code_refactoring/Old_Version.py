
# coding: utf-8

# In[1]:

def create_opt_problem(X, y, sim='correl', rel='correl', verbose=False):
    """
    % Function generates matrix Q and vector b
    % which represent feature similarities and feature relevances
    %
    % Input:
    % X - [m, n] - design matrix
    % y - [m, k] - target vector
    % sim - string - indicator of the way to compute feature similarities,
    % supported values are 'correl' and 'mi'
    % rel - string - indicator of the way to compute feature significance,
    % supported values are 'correl', 'mi'
    % 
    % Defaults are 'correl'
    %
    % Output:
    % Q - [n ,n] - matrix of features similarities
    % b - [n, k] - vector of feature relevances
    """
    X = np.array(X, dtype=np.float)
    y = np.array(y, dtype=np.float)
    
    eps = 1e-12
    
    if verbose == True:
        print("Constructing the problem...")
        print('Similarity measure: %s, feature relevance measure: %s' % (sim, rel))
    
    if len(y.shape) == 1:
        y_mat = y[:, np.newaxis]
    else:
        y_mat = y[:]
    
    n = X.shape[0]
    m = X.shape[1]
    k = y_mat.shape[1]
    
    if (sim == 'correl' or rel == 'correl'):
        together = np.hstack([X, y_mat]).T
        cor = np.corrcoef(together)
        #idxs_nz = np.where(np.sum(together ** 2, axis = 1) != 0)[0]
        #corr = np.corrcoef(together[:, idxs_nz])
        #cor = np.zeros((m + k, m + k))
        #print(idxs_nz, corr.shape)
        #for i, idx in enumerate(idxs_nz):
            #print(idxs_nz, np.hstack([X, y_mat]).T.shape, corr.shape, cor[idx, idxs_nz].shape, corr[i, :].shape)
            #cor[idx, idxs_nz] = corr[i, :]
            #cor[idx, -k:] = corr[i, -k:]
        
    
    if sim == 'correl':
        Q = cor[:-k, :-k]
    elif rel == 'mi':
        Q = np.zeros((m, m))
        for i in range(m):
            for j in range(i, m):
                Q[i][j] = fs.mutual_info_regression(X[:, i].reshape(-1, 1), X[:, j])
                Q[j][i] = Q[i][j]
    else:
        print("Wrong similarity measure")
        return
    
    if rel == 'correl':
        b = cor[:-k, -k:]
    elif rel == 'mi':
        b = np.zeros((m, k))
        for i in range(m):
            for j in range(k):
                b[i][j] = fs.mutual_info_regression(X[:, i].reshape(-1, 1), y_mat[:, j])
    else:
        print("Wrong relevance measure")
        return
    
    Q = np.nan_to_num(Q)
    b = np.nan_to_num(b)
    Q = np.abs(Q)
    b = np.abs(b).mean(axis=1)
    
    min_eig = scipy.linalg.eigh(Q)[0][0]
    if min_eig < 0:
        Q = Q - (min_eig - self.eps) * np.eye(*Q.shape)
            
    print()
        
    if verbose == True:
        print("Problem has been constructed.")
    return Q, np.abs(b)


def solve_opt_problem(Q, b, verbose=False):
    """
     Function solves the quadratic optimization problem stated to select
     significance and noncollinear features

     Input:
     Q - [n, n] - matrix of features similarities
     b - [n, 1] - vector of feature relevances

     Output:
     x - [n, 1] - solution of the quadratic optimization problem
    """
    
    n = Q.shape[0]
    x = cvx.Variable(n)
    
    objective = cvx.Minimize(cvx.quad_form(x, Q) - 1. * b.T * x)
    constraints = [x >= 0, x <= 1]
    prob = cvx.Problem(objective, constraints)
    
    if verbose == True:
        print("Solving the QP problem...")
    
    prob.solve()
    
    if verbose == True:
        print("The problem has been solved!")
        print("Problem status:", prob.status)
        print

    return np.array(x.value).flatten()
    
def quadratic_programming(X, y, sim='correl', rel='correl', verbose=False):
    Q, b = create_opt_problem(X, y, sim, rel, verbose)
    print
    qp_score = solve_opt_problem(Q, b, verbose)
    return qp_score


# In[ ]:



