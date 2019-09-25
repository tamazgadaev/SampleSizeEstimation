import numpy as np
import scipy.stats as sps
from scipy.optimize import minimize as minimize
from scipy.optimize import minimize_scalar
from tqdm import tqdm_notebook as tqdm
from sklearn.utils import shuffle as shuffle
from sklearn.metrics import mean_squared_error, f1_score, roc_auc_score, roc_curve, auc

import lib.utils as ut

import statsmodels.api as sm

def lagrange(X, y, ind_u = None, epsilon = 0.2, alpha = 0.05, beta = 0.2):
    m, n = X.shape
    statmodel = ut.define_model(y)

    if ind_u is None:
        ind_u = np.concatenate([np.ones(n // 2), np.zeros(n - n//2)]).astype(bool)
        # ind_u = np.concatenate([np.ones(1), np.zeros(n-1)]).astype(bool)

    ind_v = ind_u == False
    
    model = statmodel(y, X)
    w_hat = ut.get_params(X, y)

    mu = model.predict(w_hat)

    if len(list(set(list(y)))) == 2:
        v = mu*(1-mu)
    else:
        v = np.ones_like(y)*(mu-y).var()

    wu0 = w_hat[ind_u] + epsilon
    wv_hat = minimize(ut.fix_variables(ut.negative_func(model.loglike_fixed), wu0, ind_u), np.zeros(ind_v.sum()),
                     jac = ut.fix_variables(ut.negative_func(model.score_fixed), wu0, ind_u, 1),
                     hess = ut.fix_variables(ut.negative_func(model.hessian_fixed), wu0, ind_u, 2),
                     method = 'Newton-CG')['x']
    w_0 = ut.stitch_vectors(wu0, wv_hat, ind_u)

    I = -model.hessian_fixed(w_0)
    I_muv = I[ind_u][:,ind_v]
    I_mvv = I[ind_v][:,ind_v]
    
    Z_star = (X[:,ind_u].T - I_muv @ np.linalg.inv(I_mvv) @ X[:,ind_v].T).T
    Z_star_matrices = np.asarray([Z_star[i,None].T @ Z_star[i, None] for i in range(m)])
    
    delta = np.ones_like(y)
    mu_star = model.predict(w_0)
    
    xi_m = (((mu - mu_star)*delta[None,:]).T * Z_star).sum(0)
    Sigma_m = ((v * delta**2).reshape(-1,1,1) * Z_star_matrices).sum(0)
    
    gamma_0 = (xi_m @ np.linalg.inv(Sigma_m) @ xi_m)/m
    gamma = ut.get_gamma(ind_u, alpha, beta)
    
    m_star = np.ceil(gamma/gamma_0).astype(int)
    return {'m*': m_star}


def likelihood_ratio(X, y, ind_u=None, epsilon = 0.2, alpha = 0.05, beta = 0.2):

    m, n = X.shape
    statmodel = ut.define_model(y)
    if ind_u is None:
        ind_u = np.concatenate([np.ones(n // 2), np.zeros(n - n//2)]).astype(bool)
        # ind_u = np.concatenate([np.ones(1), np.zeros(n-1)]).astype(bool)

    model = statmodel(y, X)
    w_hat = ut.get_params(X, y)

    wu0 = w_hat[ind_u] + epsilon
    wv_hat = minimize(ut.fix_variables(ut.negative_func(model.loglike_fixed), wu0, ind_u), np.zeros(X.shape[1] - ind_u.sum()),
                         jac = ut.fix_variables(ut.negative_func(model.score_fixed), wu0, ind_u, 1),
                         hess = ut.fix_variables(ut.negative_func(model.hessian_fixed), wu0, ind_u, 2),
                         method = 'Newton-CG')['x']
    w_0 = ut.stitch_vectors(wu0, wv_hat, ind_u)

    theta = X @ w_hat
    theta_star = X @ w_0

    if len(list(set(list(y)))) == 2:
        a = 1.
        b = lambda w: -np.log(1 - model.predict(w) + 1e-30)
        grad_b = lambda w: model.predict(w)
    else:
        a = 2*((y - theta).std())**2
        b = lambda w: model.predict(w)**2
        grad_b = lambda w: 2*model.predict(w)
    


    delta_star = 2*(1/a)*((theta-theta_star)*grad_b(w_hat) - b(w_hat) + b(w_0)).mean()

    gamma_star = ut.get_gamma(ind_u, alpha, beta)

    m_star = np.ceil(gamma_star/delta_star).astype(int)
    return {'m*': m_star}



def wald(X, y, ind_u = None, epsilon = 0.2, alpha = 0.05, beta = 0.2):
    m, n = X.shape
    statmodel = ut.define_model(y)
    if ind_u is None:
        ind_u = np.concatenate([np.ones(n // 2), np.zeros(n - n//2)]).astype(bool)
        # ind_u = np.concatenate([np.ones(1), np.zeros(n-1)]).astype(bool)

    model = statmodel(y, X)
    w_hat = ut.get_params(X, y)

    wu0 = w_hat[ind_u]+ epsilon
    wv_hat = minimize(ut.fix_variables(ut.negative_func(model.loglike_fixed), wu0, ind_u), np.zeros(X.shape[1] - ind_u.sum()),
                         jac = ut.fix_variables(ut.negative_func(model.score_fixed), wu0, ind_u, 1),
                         hess = ut.fix_variables(ut.negative_func(model.hessian_fixed), wu0, ind_u, 2),
                         method = 'Newton-CG')['x']
    w_0 = ut.stitch_vectors(wu0, wv_hat, ind_u)
    V = np.array(np.linalg.inv(-model.hessian_fixed(w_hat))[ind_u][:,ind_u], ndmin=2)
    V_star = np.array(np.linalg.inv(-model.hessian_fixed(w_0))[ind_u][:,ind_u], ndmin=2)
    Sigma = m*V
    Sigma_star = m*V_star
    w_u = w_hat[ind_u]
    w_u0 = w_0[ind_u]

    delta = np.dot((w_u - w_u0), np.linalg.inv(Sigma)@(w_u - w_u0))
    
    classification = len(list(set(list(y)))) == 2
    
    if classification:
        alpha_star = ut.fix_alpha(alpha, Sigma, Sigma_star)
    else:
        alpha_star = alpha
    gamma_star = ut.get_gamma(ind_u, alpha_star, beta)
    m_star = np.ceil(gamma_star/delta).astype(int)
    return {'m*':m_star}

def cross_val(X, y, k_for_meaning = 100, epsilon = 0.05, begin = None, end = None, num = None, progress_bar = None, multiprocess = False):
    
    if progress_bar is None:
        progress_bar = list
    else:
        progress_bar=lambda x: tqdm(x, leave = False)
    if end is None:
        end = X.shape[0] - 1
    if begin is None:
        begin = 2*X.shape[1]
    if num is None:
        num = 5

    subset_sizes = np.arange(begin, end, num, dtype=np.int64)
    m_size = end

    X, y = shuffle(X, y)
    
    list_of_E, list_of_S = ut.calculater(X, y, subset_sizes, k_for_meaning, ut.RS, multiprocess, progress_bar)

    for m, mean in zip(reversed(subset_sizes), reversed(list_of_E)):
        if mean < epsilon:
            m_size = m

    return {'m*': m_size,
            'E': np.array(list_of_E),
            'S': np.array(list_of_S),
            'm': np.array(subset_sizes),
           }

def APVC(X, y, k_for_meaning = 1000, epsilon = 0.5, begin = None, end = None, num = None, progress_bar = None, multiprocess = False):
    if progress_bar is None:
        progress_bar = list
    else:
        progress_bar=lambda x: tqdm(x, leave = False)
    if end is None:
        end = X.shape[0] - 1
    if begin is None:
        begin = 2*X.shape[1]
    if num is None:
        num = 5

    subset_sizes = np.arange(begin, end, num, dtype=np.int64)
    m_size = end

    X, y = shuffle(X, y)

    list_of_E, list_of_S = ut.calculater(X, y, subset_sizes, k_for_meaning, ut.hDispersion, multiprocess, progress_bar)

    for m, mean, std in zip(reversed(subset_sizes), reversed(list_of_E), reversed(list_of_S)):
        if mean < epsilon:
            m_size = m

    return {'m*': m_size,
            'E': np.array(list_of_E),
            'S': np.array(list_of_S),
            'm': np.array(subset_sizes),
           }

def ACC(X, y, k_for_meaning = 100, l = 0.25, alpha = 0.05, begin = None, end = None, num = None, progress_bar = None, multiprocess = False):
    if progress_bar is None:
        progress_bar = list
    else:
        progress_bar=lambda x: tqdm(x, leave = False)
    if end is None:
        end = X.shape[0] - 1
    if begin is None:
        begin = 2*X.shape[1]
    if num is None:
        num = 5

    subset_sizes = np.arange(begin, end, num, dtype=np.int64)
    m_size = end

    X, y = shuffle(X, y)

    list_of_E, list_of_S = ut.calculater(X, y, subset_sizes, k_for_meaning, ut.iDistribution, multiprocess, progress_bar, l)

    for m, mean, std in zip(reversed(subset_sizes), reversed(list_of_E), reversed(list_of_S)):
        if mean > 1 - alpha:
            m_size = m

    return {'m*': m_size,
            'E': np.array(list_of_E),
            'S': np.array(list_of_S),
            'm': np.array(subset_sizes),
           }


def ALC(X, y, k_for_meaning = 100, l = 0.5, alpha = 0.05, begin = None, end = None, num = None, progress_bar = None, multiprocess = False):
    if progress_bar is None:
        progress_bar = list
    else:
        progress_bar=lambda x: tqdm(x, leave = False)
    if end is None:
        end = X.shape[0] - 1
    if begin is None:
        begin = 2*X.shape[1]
    if num is None:
        num = 5

    subset_sizes = np.arange(begin, end, num, dtype=np.int64)
    m_size = end

    X, y = shuffle(X, y)

    list_of_E, list_of_S = ut.calculater(X, y, subset_sizes, k_for_meaning, ut.aDistribution, multiprocess, progress_bar, alpha)

    for m, mean, std in zip(reversed(subset_sizes), reversed(list_of_E), reversed(list_of_S)):
        if mean < l:
            m_size = m

    return {'m*': m_size,
            'E': np.array(list_of_E),
            'S': np.array(list_of_S),
            'm': np.array(subset_sizes),
           }

def MAX_U(X, y, k_for_meaning = 100, c = 0.005, begin = None, end = None, num = None, progress_bar = None, multiprocess=False):
    if progress_bar is None:
        progress_bar = list
    else:
        progress_bar=lambda x: tqdm(x, leave = False)
    if end is None:
        end = X.shape[0] - 1
    if begin is None:
        begin = 2*X.shape[1]
    if num is None:
        num = max(5, int(X.shape[0]/20))

    subset_sizes = np.arange(begin, end, num, dtype=np.int64)
    m_size = end

    X, y = shuffle(X, y)

    list_of_E, list_of_S = ut.calculater(X, y, subset_sizes, k_for_meaning, ut.uFunction, multiprocess, progress_bar, c)
    return {'m*': subset_sizes[np.argmax(np.array(list_of_E))],
            'E': np.array(list_of_E),
            'S': np.array(list_of_S),
            'm': np.array(subset_sizes),
           }

def KL_method(X, y, k_for_meaning = 5, epsilon = 0.01, begin = None, end = None, num = None, progress_bar = None, multiprocess=False):
    if progress_bar is None:
        progress_bar = list
    else:
        progress_bar=lambda x: tqdm(x, leave = False)
    if end is None:
        end = X.shape[0] - 1
    if begin is None:
        begin = 2*X.shape[1]
    if num is None:
        num = 5

    subset_sizes = np.arange(begin, end, num, dtype=np.int64)
    m_size = end

    X, y = shuffle(X, y)

    list_of_E, list_of_S = ut.calculater(X, y, subset_sizes, k_for_meaning, ut.klFunction, multiprocess, progress_bar)

    for m, mean in zip(reversed(subset_sizes), reversed(list_of_E)):
        if mean < epsilon:
            m_size = m

    return {'m*': m_size,
            'E': np.array(list_of_E),
            'S': np.array(list_of_S),
            'm': np.array(subset_sizes),
           }


def bootstrap(X, y, k_for_meaning = 1000, epsilon = 0.5, begin = None, end = None, num = None, progress_bar = None, multiprocess=False):
    if progress_bar is None:
        progress_bar = list
    else:
        progress_bar=lambda x: tqdm(x, leave = False)
    if end is None:
        end = X.shape[0] - 1
    if begin is None:
        begin = 2*X.shape[1]
    if num is None:
        num = 5

    subset_sizes = np.arange(begin, end, num, dtype=np.int64)
    m_size = end

    X, y = shuffle(X, y)

    list_of_E = []
    list_of_S = []

    for m in progress_bar(subset_sizes):
        list_of_W = []
        for _ in progress_bar(range(k_for_meaning)):
            X_m, y_m = ut.get_subset(X, y, m)
            list_of_W.append(ut.bFunction(X_m, y_m))

        list_of_W = np.array(list_of_W)
        params_ci = np.zeros([X.shape[1], 2])

        for i in range(X.shape[1]):
            params_ci[i, 0] = np.percentile(list_of_W[:,i], 2.5)
            params_ci[i, 1] = np.percentile(list_of_W[:,i], 97.5)

        lengths = np.abs(params_ci[:,1] - params_ci[:,0])


        mean = np.max(lengths)
        std = 0


        list_of_E.append(mean)
        list_of_S.append(std)

    for m, mean in zip(reversed(subset_sizes), reversed(list_of_E)):
        if mean < epsilon:
            m_size = m

    return {'m*': m_size,
            'E': np.array(list_of_E),
            'S': np.array(list_of_S),
            'm': np.array(subset_sizes),
           }

def LogisticRegressionMethod(X, y, ind = 0, alpha = 0.05, beta = 0.2):
    statmodel = ut.define_model(y)


    w_hat0 = ut.get_params(np.delete(X, ind, axis = 1), y)
    w_hat1 = ut.get_params(X, y)


    predict0 = statmodel(y, np.delete(X, ind, axis = 1)).predict(w_hat0)
    predict1 = statmodel(y, X).predict(w_hat1)
    
     
    fpr0, tpr0, threshold0 = roc_curve(y, predict0)
    fpr1, tpr1, threshold1 = roc_curve(y, predict1)
    
    c0 = threshold0[np.argmax((tpr0 - threshold0)**2 - (fpr0 - threshold0)**2)]
    c1 = threshold1[np.argmax((tpr1 - threshold1)**2 - (fpr1 - threshold1)**2)]

    p0 = np.mean(predict0 > c0)
    p1 = np.mean(predict1 > c0)
    
    t_alpha = sps.norm.ppf(1 - 0.5*alpha)
    t_beta = sps.norm.ppf(1 - beta)
    m_size = ((np.sqrt(p0*(1-p0))*t_alpha+t_beta*np.sqrt(p1*(1-p1)))**2)/((p0-p1)**2)

    return {'m*': int(m_size),
           } 

