import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
import scipy.stats as sps
import scipy.special as sp
from scipy.optimize import minimize as minimize
from scipy.optimize import minimize_scalar as minimize_scalar
from tqdm import tqdm_notebook as tqdm

from sklearn.model_selection import train_test_split
import functools
import os as os
import pickle as pickle
import dill
from sklearn.metrics import mean_squared_error as mean_squared_error
from sklearn.metrics import roc_auc_score as roc_auc_score
from statsmodels.stats import multitest
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.discrete.discrete_model import Logit
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

from scipy.special import expit as expit

from sklearn.preprocessing import StandardScaler, scale

from sklearn.linear_model import LogisticRegression as LogisticRegression
from sklearn.linear_model import Ridge as Ridge
from sklearn.metrics import mean_absolute_error as mean_absolute_error

import statsmodels.api as sm

from multiprocessing import Pool
import math
from scipy.linalg import fractional_matrix_power



class LinearModel():
    def __init__(self, y, X, alpha = 0.01):
        self.y = y
        self.X = X
        self.alpha = alpha
        self.w = None

        self.prior = sps.multivariate_normal(mean = np.zeros(X.shape[1]), cov = alpha*np.eye(X.shape[1]))
        self.n = X.shape[1]
        self.m = y.shape[0]
        self.log2pi = np.log(2*np.pi)

    def fit(self):
        self.w = np.linalg.inv(self.alpha*np.eye(self.n) + self.X.T@self.X)@self.X.T@self.y
        return self.w

    def predict(self, params, X = None):
        if X is None:
            X = self.X
        return X@params

    def loglike(self, params):
        return 0.5*(-np.sum((self.y - self.X@params)**2) - self.m*self.log2pi)

    def score(self, params):
        return self.X.T@self.y - self.X.T@self.X@params

    def hessian(self, params):
        return -self.X.T@self.X

    def loglike_fixed(self, params):
        return self.loglike(params) + self.prior.logpdf(params)

    def score_fixed(self, params):
        return self.score(params) - self.alpha*params

    def hessian_fixed(self, params):
        return self.hessian(params) - self.alpha*np.eye(self.n)
    
    def covariance(self, params):
        return np.linalg.inv(-self.hessian_fixed(params))


class LogitModel():
    def __init__(self, y, X, alpha = 0.01):
        self.y = y
        self.X = X
        self.alpha = alpha
        self.w = None

        self.prior = sps.multivariate_normal(mean = np.zeros(X.shape[1]), cov = alpha*np.eye(X.shape[1]))

        self.n = X.shape[1]
        self.m = y.shape[0]

    def fit(self):
        model_sk_learn = LogisticRegression(C = 1./self.alpha)
        model_sk_learn.fit(self.X, self.y)
        self.w = model_sk_learn.coef_[0]
        return self.w

    def predict(self, params, X = None):
        if X is None:
            X = self.X
        return expit(X@params)

    def loglike(self, params):
        epsilon = 10**(-10)
        q = 2*self.y - 1
        res = expit(q*np.dot(self.X, params))
        res = res + (res < epsilon)*epsilon
        return np.sum(np.log(res))

    def score(self, params):
        theta = expit(self.X@params)
        return np.dot(self.y - theta, self.X)

    def hessian(self, params):
        theta = expit(self.X@params)
        return -np.dot(theta*(1-theta)*self.X.T, self.X)

    def loglike_fixed(self, params):
        return self.loglike(params) + self.prior.logpdf(params)

    def score_fixed(self, params):
        return self.score(params) - self.alpha*params

    def hessian_fixed(self, params):
        return self.hessian(params) - self.alpha*np.eye(self.n)

    def covariance(self, params):
        return np.linalg.inv(-self.hessian_fixed(params))


class multicalculated():
    def __init__(self, X, y, function, *kwargs):
        self.X = X
        self.y = y
        self.function = function
        self.kwargs = kwargs
    
    def calc(self, m):
        X_m, y_m = get_subset(self.X, self.y, m)
        if len(self.kwargs) > 0:
            return self.function(X_m, y_m, self.kwargs[0])
        else:
            return self.function(X_m, y_m)

def calculater(X, y, subset_sizes, k_for_meaning, function, multiprocess, progress_bar, *kwargs):

    if multiprocess == True:
        list_of_answers = []
        pool = Pool()

        if len(kwargs) > 0:
            multi = multicalculated(X, y, function, kwargs[0])
        else:
            multi = multicalculated(X, y, function)

        func = multi.calc

        # for _ in progress_bar(range(k_for_meaning)):
        points_one = np.ones(k_for_meaning, dtype = np.int64)
        for m in progress_bar(subset_sizes):
            list_of_answers.append(np.asarray(pool.map(func, m*points_one)))

        list_of_answers = np.asarray(list_of_answers)

        list_of_E = np.mean(list_of_answers, axis = 1)
        list_of_S = np.std(list_of_answers, axis = 1)
        pool.close()
        pool.join()

    else:
        list_of_E = []
        list_of_S = []
        fun = None
        if len(kwargs) > 0:
            fun = lambda X, y : function(X, y, kwargs[0])
        else:
            fun = function

        for m in progress_bar(subset_sizes):
            list_of_D = []
            for _ in progress_bar(range(k_for_meaning)):
                X_m, y_m = get_subset(X, y, m)
                list_of_D.append(fun(X_m, y_m))
            list_of_D = np.asarray(list_of_D)


            list_of_E.append(np.mean(list_of_D))
            list_of_S.append(np.std(list_of_D))

    return list_of_E, list_of_S


def define_model(y):
    if len(list(set(list(y)))) == 2:
        model = lambda y, X: LogitModel(y, X, alpha = 0.01)
    else:
        model = lambda y, X: LinearModel(y, X, alpha = 0.01)
    return model

def get_params(X, y):
    if len(list(set(list(y)))) == 2:
        model_sk_learn = LogisticRegression(C = 100)
        model_sk_learn.fit(X, y)
        w_hat = model_sk_learn.coef_[0]
    else:
        w_hat = np.linalg.inv(0.01*np.eye(X.shape[1]) + X.T@X)@X.T@y
    return w_hat

def RS(X, y):
    statmodel = define_model(y)

    X_train, X_test, y_train, y_test = train_test_split_safe(X, y)

    w_hat = get_params(X_train, y_train)

    S_train = statmodel(y_train, X_train).loglike(w_hat)/y_train.shape[0]
    S_test = statmodel(y_test, X_test).loglike(w_hat)/ y_test.shape[0]

    return S_train - S_test


def hDispersion(X, y):
    statmodel = define_model(y)

    w_hat = get_params(X, y)
    cov = np.linalg.inv(0.01*np.eye(w_hat.shape[0]) - statmodel(y, X).hessian(w_hat))

    return np.sqrt(np.sum((np.linalg.eigvals(cov)/2)**2))
    


def iDistribution(X, y, l = 0.5):
    statmodel = define_model(y)

    w_hat = get_params(X, y)
    cov = np.linalg.inv(0.01*np.eye(w_hat.shape[0]) - statmodel(y, X).hessian(w_hat))

    W = sps.multivariate_normal(mean=np.zeros(w_hat.shape[0]), cov = cov).rvs(size=1000)
    return (np.sqrt((W**2).sum(axis=1)) < 3*l).mean()



def aDistribution(X, y, alpha = 0.05):
    statmodel = define_model(y)

    w_hat = get_params(X, y)
    cov = np.linalg.inv(0.01*np.eye(w_hat.shape[0]) - statmodel(y, X).hessian(w_hat))

    W = sps.multivariate_normal(mean=np.zeros(w_hat.shape[0]), cov = cov).rvs(size=1000)

    function = lambda r: np.abs( (np.sqrt((W**2).sum(axis=1)) > 3*r).mean() - alpha)

    res = minimize_scalar(function, bounds=(0.01, 1), method='Bounded', options={'maxiter':10})['x']
    return res

def uFunction(X, y, c = 0.005):
    statmodel = define_model(y)

    prior = sps.multivariate_normal(mean = np.zeros(X.shape[1]), cov = 0.01*np.eye(X.shape[1]))
    model = statmodel(y, X)

    w_hat = get_params(X, y)

    cov = np.linalg.inv(0.01*np.eye(X.shape[1]) - statmodel(y, X).hessian(w_hat))

    W = sps.multivariate_normal(mean=w_hat, cov = cov).rvs(size=100)

    u = []
    for w in W:
        u.append(model.loglike(w) + prior.logpdf(w))

    return np.mean(u)/y.shape[0]  - c*y.shape[0]


def D_KL_normal(m_0, cov_0, m_1, cov_1, cov_0_inv, cov_1_inv):
    m_0 = np.array(m_0, ndmin=1)
    m_1 = np.array(m_1, ndmin=1)
    cov_0 = np.array(cov_0, ndmin=2)
    cov_1 = np.array(cov_1, ndmin=2)
    
    D_KL_1 = np.sum(np.diagonal(cov_1@cov_0_inv))
    D_KL_2 = float(np.reshape((m_1 - m_0), [1, -1])@cov_1@np.reshape((m_1 - m_0), [-1, 1]))
    D_KL_3 = -m_0.shape[0]
    D_KL_4 = float(np.log(np.linalg.det(cov_0)/np.linalg.det(cov_1)))
    
    return 0.5*(D_KL_1 + D_KL_2 + D_KL_3 + D_KL_4)

def klFunction(X, y):
    statmodel = define_model(y)

    model_0 = statmodel(y, X)
    m_0 = get_params(X, y)
    cov_0_inv = 0.01*np.eye(m_0.shape[0]) - model_0.hessian(m_0)
    cov_0 = np.linalg.inv(cov_0_inv)

    # ind = np.random.randint(0, X.shape[0])
    indexes = np.random.permutation(X.shape[0])[:50]

    list_of_res = []

    for ind in indexes:
        X_new = np.delete(X, ind, axis = 0)
        y_new = np.delete(y, ind, axis = 0)

        model_1 = statmodel(y_new, X_new)
        m_1 = get_params(X_new, y_new)
        cov_1_inv = 0.01*np.eye(m_1.shape[0]) - model_1.hessian(m_1)
        cov_1 = np.linalg.inv(cov_1_inv)
        list_of_res.append(D_KL_normal(m_0, cov_0, m_1, cov_1, cov_0_inv, cov_1_inv))

    return np.mean(list_of_res)

def bFunction(X, y):

    statmodel = define_model(y)
    
    w_hat = get_params(X, y)
    
    if len(list(set(list(y)))) != 2:
        y_hat = statmodel(y, X).predict(w_hat)
        Es = y - y_hat
        y_new = y_hat + (Es - Es.mean())
        w_res = get_params(X, y_new)
    else:
        w_res = w_hat
    return w_res

def get_synthetic_sample(m = 1000, n = 10, classification = True):
    w = np.random.randn(n)
    X = np.random.randn(m, n)
    if classification:
        p = expit(X@np.reshape(w, [-1, 1]))
        y = sps.bernoulli(p).rvs().reshape(-1)
    else:
        y = X @ w + np.random.normal(size = m) + np.random.randn(1)
    return X, y, w

def get_subset(X, y, M, duplications = True):
    if M < 4:
        M = 4
    if duplications:
        indexes = np.random.randint(low = 0, high=X.shape[0], size = M)
    else:
        indexes = np.random.permutation(X.shape[0])[:M]
    
    X_M = X[indexes, :]
    y_M = y[indexes]
    while ((y_M == 0).sum() > M - 2 or (y_M == 1).sum() > M - 2):
        if duplications:
            indexes = np.random.randint(low = 0, high=X.shape[0], size = M)
        else:
            indexes = np.random.permutation(X.shape[0])[:M]
    
        X_M = X[indexes, :]
        y_M = y[indexes]
    return X_M, y_M

def train_test_split_safe(X, y, test_size = 0.5):
    M = int(X.shape[0]*test_size)
    indexes_test = np.random.permutation(X.shape[0])[:M]
    indexes_train = np.random.permutation(X.shape[0])[M:]
    X_train = X[indexes_train, :]
    X_test = X[indexes_test, :]
    y_train = y[indexes_train]
    y_test = y[indexes_test]
    while ((y_train == 0).all() or (y_train == 1).all() or (y_test == 0).all() or (y_test == 1).all()):
        indexes_test = np.random.permutation(X.shape[0])[:M]
        indexes_train = np.random.permutation(X.shape[0])[M:]
        X_train = X[indexes_train, :]
        X_test = X[indexes_test, :]
        y_train = y[indexes_train]
        y_test = y[indexes_test]
    return X_train, X_test, y_train, y_test

# Построение ошибки от размера выборки
def get_error_track(X, y, test_train_proportional = 0.25, sample_sizes = None, k_for_meaning = 100, statmodel = None, progress_bar = None):
    
    if progress_bar is None:
        progress_bar = list

    if sample_sizes is None:
        sample_sizes = np.linspace(2*X.shape[1], X.shape[0] - 1, num = 100, dtype=np.int64)
    
    if statmodel is None:
        statmodel = define_model(y)

    list_of_error_mean = []
    list_of_error_std = []
    
	# X_train, X_test, y_train, y_test = train_test_split_safe(X_M, y_M, test_size = test_train_proportional)

    for size in progress_bar(sample_sizes):
        list_of_error_for_k = []
        
        for _ in progress_bar(range(k_for_meaning)):
            X_train, X_test, y_train, y_test = train_test_split_safe(X, y, test_train_proportional)
            X_M, y_M = get_subset(X_train, y_train, size)
            X_m, y_m = get_subset(X_test, y_test, size)

            w_hat = get_params(X_M, y_M)
            # w_hat = statmodel(y_M, X_M).fit_regularized(alpha = 0.01, L1_wt = 0.0).params
            list_of_error_for_k.append(statmodel(y_m, X_m).loglike(w_hat)/y_m.shape[0])
            
        list_of_error_mean.append(np.mean(list_of_error_for_k))
        list_of_error_std.append(np.std(list_of_error_for_k))
        
    return np.array(list_of_error_mean), np.array(list_of_error_std)

# Построение m* для разных размеров выборок

def get_m_size_track(X, y, sample_sizes, k_for_meaning = 100, method = None, progress_bar = None):

    if progress_bar is None:
        progress_bar = list

    list_of_m_size_mean = []
    list_of_m_size_std = []
    pool = Pool(4)

    if method is None:
        return None, None

    
    for M in progress_bar(sample_sizes):
        list_of_res = []
        for _ in progress_bar(range(k_for_meaning)):
            X_M, y_M = get_subset(X, y, M)
            res = method(X_M, y_M)
            list_of_res.append(res['m*'])
        list_of_m_size_mean.append(np.mean(list_of_res))
        list_of_m_size_std.append(np.std(list_of_res))
        
    return np.array(list_of_m_size_mean), np.array(list_of_m_size_std)

def experiment_for_datasets(datasets, k_for_meaning = 100, test_train_proportional = 0.25, progress_bar = None): 
    if progress_bar is None:
        progress_bar = list

    list_of_answers = []
    for dataset in datasets:
        data = dataset.copy()
        name = data['name']
        X = data['X']
        y = data['y']
        folder_path = data['folder_path']
        if os.path.exists(folder_path) == False:
            os.mkdir(folder_path)
        backup = data['backup']

        sample_sizes = data['sample_sizes']

        methods = data['methods']

        methods_list = list(methods.keys()).copy()

        
        # X_train, X_test, y_train, y_test = train_test_split_safe(X, y, test_size = test_train_proportional)
        print('Dataset ' + name + ' begining:')
        answers = {}

        is_use_backup, path_backup = [False, None]

        if 'likelihood' in methods_list:
            method = methods['likelihood']
            methods_sample_sizes = sample_sizes
            if 'likelihood' in backup.keys():
                is_use_backup, path_backup = backup['likelihood']
                path_backup = folder_path + path_backup

            if is_use_backup == False:
                print('Log-likelihood calculating:')
                s_mean, s_std = method(X, y, sample_sizes=methods_sample_sizes)

                if path_backup is not None:
                    pickle.dump([s_mean, s_std, methods_sample_sizes], open(path_backup, "wb"))
                print('Log-likelihood calculated.')
            else:
                if os.path.exists(path_backup):
                    s_mean_loaded , s_std_loaded, methods_sample_sizes_loaded = pickle.load(open(path_backup, "rb"))
                    print('Log-likelihood loaded.')

                    methods_sample_sizes_calc = [x for x in methods_sample_sizes if x not in methods_sample_sizes_loaded]

                    print('Log-likelihood calculating:')
                    s_mean_calc, s_std_calc = method(X, y, sample_sizes=methods_sample_sizes)

                    methods_sample_sizes_all = np.hstack([methods_sample_sizes_loaded, methods_sample_sizes_calc])
                    
                    s_mean_all = np.hstack([s_mean_loaded, s_mean_calc])
                    s_std_all = np.hstack([s_std_loaded, s_std_calc])

                    methods_sample_sizes = methods_sample_sizes_all[np.argsort(methods_sample_sizes_all)]
                    s_mean = s_mean_all[np.argsort(methods_sample_sizes_all)]
                    s_std = s_std_all[np.argsort(methods_sample_sizes_all)]

                    if path_backup is not None:
                        pickle.dump([s_mean, s_std, methods_sample_sizes], open(path_backup, "wb"))
                    print('Log-likelihood calculated.')

                else:
                    print('Log-likelihood calculating:')
                    s_mean, s_std = method(X, y, sample_sizes=methods_sample_sizes)

                    if path_backup is not None:
                        pickle.dump([s_mean, s_std, methods_sample_sizes], open(path_backup, "wb"))
                    print('Log-likelihood calculated.')

            answers['likelihood'] = [s_mean, s_std, methods_sample_sizes]
            methods_list.remove('likelihood')


        for key in methods_list:
            methods_sample_sizes = sample_sizes
            method = methods[key]

            is_use_backup, path_backup = [False, None]
            if key in backup.keys():
                is_use_backup, path_backup = backup[key]
                path_backup = folder_path + path_backup

            if is_use_backup == False:
                print('Sample sizes calculating for ' + key +':')
                m_mean, m_std = get_m_size_track(X, y, methods_sample_sizes, 
                                                    k_for_meaning = k_for_meaning, progress_bar=progress_bar, 
                                                    method = method)
                if path_backup is not None:
                    pickle.dump([m_mean, m_std, methods_sample_sizes], open(path_backup, "wb"))
                print('Sample sizes calculated for ' + key +'.')
            else:
                if os.path.exists(path_backup):
                    m_mean_loaded , m_std_loaded, methods_sample_sizes_loaded = pickle.load(open(path_backup, "rb"))
                    print('Sample sizes loaded for ' + key +'.')

                    methods_sample_sizes_calc = [x for x in methods_sample_sizes if x not in methods_sample_sizes_loaded]

                    print('Sample sizes calculating for ' + key +':')
                    m_mean_calc, m_std_calc = get_m_size_track(X, y,  methods_sample_sizes_calc, 
                                                    k_for_meaning = k_for_meaning, progress_bar=progress_bar, 
                                                    method = method)

                    methods_sample_sizes_all = np.hstack([methods_sample_sizes_loaded, methods_sample_sizes_calc])
                    
                    m_mean_all = np.hstack([m_mean_loaded, m_mean_calc])
                    m_std_all = np.hstack([m_std_loaded, m_std_calc])

                    methods_sample_sizes = methods_sample_sizes_all[np.argsort(methods_sample_sizes_all)]
                    m_mean = m_mean_all[np.argsort(methods_sample_sizes_all)]
                    m_std = m_std_all[np.argsort(methods_sample_sizes_all)]

                    if path_backup is not None:
                        pickle.dump([m_mean, m_std, methods_sample_sizes], open(path_backup, "wb"))
                    print('Sample sizes calculated for ' + key +'.')

                else:
                    print('Sample sizes calculating for ' + key +':')
                    m_mean, m_std = get_m_size_track(X, y, methods_sample_sizes, 
                                                        k_for_meaning = k_for_meaning, progress_bar=progress_bar, 
                                                        method = method)
                    if path_backup is not None:
                        pickle.dump([m_mean, m_std, methods_sample_sizes], open(path_backup, "wb"))
                    print('Sample sizes calculated for ' + key +'.')
            
            answers[key] = [m_mean, m_std, methods_sample_sizes]
            
        data['answers'] = answers
        
        list_of_answers.append(data)
    
    return list_of_answers


def LoadExperiment(folder_path = None, backup = None):
    if backup is None:
        backup = { 'likelihood': "likelihood.p",
                   'cross_val': "cross_val.p",
                   'lagrange': "lagrange.p",
                   'likelihood_ratio': "likelihood_ratio.p",
                   'wald': "wald.p",
                   'apvc': "apvc.p",
                   'acc': "acc.p",
                   'alc': "alc.p",
                   'use': "use.p",
                   'dkl': "dkl.p",
                 }
        
    if folder_path is None:
        return None
    if os.path.exists(folder_path) == False:
        return None
    
    data = {}
    
    for key in backup.keys():
        data[key] = pickle.load(open(folder_path + backup[key], "rb"))
        
    return data

def plot_graph(data = None, methods = None, name = None, folder_path = 'graphic/'):
    """
    methods:
        'cross_val' --- 
        'lagrange' ---
        'likelihood_ratio' ---
        'wald'
        'apvc'
        'acc'
        'alc'
        'use'
    """
    if data is None:
        return None
    if methods is None:
        methods = list(data.keys()).copy()

        methods.remove('likelihood')
        
    mean, std, sample_sizes = data['likelihood']

    plt.plot(sample_sizes, mean)
    plt.fill_between(sample_sizes, mean-std, mean+std, alpha=0.2)

    plt.xlabel('Sample size')
    plt.ylabel('log-likelihood')
    plt.grid()

    if name is not None:
        plt.savefig(folder_path+name+"_likelihood.pdf")
    plt.show()


    if len(methods) > 0:
        for key in methods:
            mean, std, sample_sizes = data[key]
            count = sum(mean+std+1 == sample_sizes)
            mean = mean[count:]
            sample_sizes = sample_sizes[count:]
            std = std[count:]
            plt.plot(sample_sizes, mean, label = key)
            plt.fill_between(sample_sizes, mean-std, mean+std, alpha=0.2)


        plt.legend(loc = 'best')
        plt.xlabel('Sample size')
        plt.ylabel('m*')
        plt.grid()

        if name is not None:
            plt.savefig(folder_path+name+"_SampleSize.pdf")
        plt.show()
    return


# =======================================================
# =======================================================
# =======================================================
# =======================================================


def negative_func(f):
    negative_func_fx = lambda x, *args: -f(x, *args)
    negative_func_f = lambda x, *args: negative_func_fx(x, *args)
    return negative_func_f

def stitch_vectors(x1, x2, ind_1):
    x = np.zeros(ind_1.size)
    x[ind_1] = x1
    x[ind_1 == False] = x2
    return x

def fix_variables(f, x1, ind_1, dim = 0):
    ind_2 = (ind_1 == False)
    if dim == 0:
        return lambda x2: f(stitch_vectors(x1, x2, ind_1))
    if dim == 1: 
        return lambda x2: f(stitch_vectors(x1, x2, ind_1))[ind_2]
    if dim == 2:
        return lambda x2: f(stitch_vectors(x1, x2, ind_1))[ind_2][:,ind_2]

def get_gamma(ind_u, alpha, beta):
    k = ind_u.sum()
    f = lambda x: np.abs(sps.chi2(k, loc=x).ppf(beta) - sps.chi2(k).ppf(1-alpha))
    gamma = minimize(f, 0.)['x'][0]
    return gamma

def fix_alpha(alpha, Sigma, Sigma_star):
    p = Sigma.shape[0]
    Sigma_12 = fractional_matrix_power(Sigma, 0.5)

    matrix = Sigma_12.T @ np.linalg.inv(Sigma_star) @ Sigma_12

    lambdas = np.real(np.linalg.eigvals(matrix))
    factorials = [1, 1, 2, 8]
    k = np.asarray([factorials[r] * np.sum(lambdas**r) for r in [1,1,2,3]])

    t1 = 4*k[1]*k[2]**2 + k[3]*(k[2]-k[1]**2)
    t2 = k[3]*k[1] - 2*k[2]**2
    chi_quantile = sps.chi2(p).ppf(1-alpha)
    if t1 < 10**(-5):
        a_new = 2 + (k[1]**2)/(k[2]**2)
        b_new = (k[1]**3)/k[2] + k[1]
        s1 = 2*k[1]*(k[3]*k[1] + k[2]*k[1]**2 - k[2]**2)
        s2 = 3*t2 + 2*k[2]*(k[2] + k[1]**2)
        alpha_star = 1 - sps.invgamma(a_new, scale = b_new).cdf(chi_quantile)
    elif t2 < 10**(-5):
        a_new = (k[1]**2)/k[2]
        b_new = k[2]/k[1]
        alpha_star = 1 - sps.gamma(a_new, scale = b_new).cdf(chi_quantile)
    else:
        a1 = 2*k[1]*(k[3]*k[1] + k[2]*k[1]**2 - k[2]**2)/t1
        a2 = 3 + 2*k[2]*(k[2] + k[1]**2)/t2
        alpha_star = 1 - sps.f(2*a1, 2*a2).cdf(a2*t2*chi_quantile/(a1*t1))
        
    return alpha_star

def LoadData(url = "https://raw.githubusercontent.com/ttgadaev/SampleSize/master/datasets/servo.csv",
             path = None):
    """
    return X, y
    """
    if path is not None:
        if os.path.exists(path):
            data = pd.read_csv(path, header=0)
        else:
            data = pd.read_csv(url, header=0)
            data.to_csv(path, header=True, index=False)
    else:
        data = pd.read_csv(url, header=0)

    y = data['answer'].values
    del data['answer']
    X = data.values
    return X, y

def DataLoader(name = "servo", is_binary_answ = True, folder_path = 'datasets/'):
    """
    1. boston reg
    2. servo reg
    3. wine class
    4. diabetes reg
    5. iris class
    6. forestfires reg
    7. synthetic1 reg
    8. synthetic2 class
    9. abalone class
    10. nba class
    11.
    12. 
    
    """
   
    if name == 'servo':
        X, y = LoadData(url = "https://raw.githubusercontent.com/ttgadaev/SampleSize/master/datasets/servo.csv", 
                        path = folder_path + 'servo.csv')

        X[np.where(X == 'A')] = 1
        X[np.where(X == 'B')] = 2
        X[np.where(X == 'C')] = 3
        X[np.where(X == 'D')] = 4
        X[np.where(X == 'E')] = 5
        X = np.array(X, dtype = np.float64)
            
    if name == 'boston':
        X, y = LoadData(url = "https://raw.githubusercontent.com/ttgadaev/SampleSize/master/datasets/boston.csv", 
                        path = folder_path + 'boston.csv')

        X = np.array(X, dtype = np.float64)
        y = np.array(y, dtype = np.float64)

    if name == 'wine':
        X, y = LoadData(url = "https://raw.githubusercontent.com/ttgadaev/SampleSize/master/datasets/wine.csv", 
                        path = folder_path + 'wine.csv')

        X = np.array(X, dtype = np.float64)
        y = np.array(y, dtype = np.int64)
        if is_binary_answ:
            X = np.delete(X, np.where(y == 2), axis = 0)
            y = np.delete(y, np.where(y == 2), axis = 0)
        
    if name == 'diabetes':
        X, y = LoadData(url = "https://raw.githubusercontent.com/ttgadaev/SampleSize/master/datasets/diabetes.csv", 
                        path = folder_path + 'diabetes.csv')

        X = np.array(X, dtype = np.float64)
        y = np.array(y, dtype = np.float64)
        
    if name == 'iris':
        X, y = LoadData(url = "https://raw.githubusercontent.com/ttgadaev/SampleSize/master/datasets/iris.csv", 
                        path = folder_path + 'iris.csv')

        X = np.array(X, dtype = np.float64)
        y = np.array(y, dtype = np.int64)
        if is_binary_answ:
            X = np.delete(X, np.where(y == 2), axis = 0)
            y = np.delete(y, np.where(y == 2), axis = 0)

    if name == 'nba':
        X, y = LoadData(url = "https://raw.githubusercontent.com/ttgadaev/SampleSize/master/datasets/nba.csv", 
                        path = folder_path + 'nba.csv')

        indexes = np.where(np.isnan(X) == True)[0]
        X = np.delete(X, indexes, axis = 0)
        y = np.delete(y, indexes, axis = 0)
        X = np.array(X, dtype = np.float64)
        y = np.array(y, dtype = np.int64)

    if name == 'abalone':
        X, y = LoadData(url = "https://raw.githubusercontent.com/ttgadaev/SampleSize/master/datasets/abalone.csv", 
                        path = folder_path + 'abalone.csv')

        X = np.array(X, dtype = np.float64)
        y = np.array(y, dtype = np.int64)
        if is_binary_answ:
            X = np.delete(X, np.where(y == 2), axis = 0)
            y = np.delete(y, np.where(y == 2), axis = 0)
            
    if name == 'forestfires':
        X, y = LoadData(url = "https://raw.githubusercontent.com/ttgadaev/SampleSize/master/datasets/forestfires.csv",
                        path = folder_path + 'forestfires.csv')
        
        list_of_month = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
        list_of_day = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

        for i, mon in enumerate(list_of_month):
            X[np.where(X == mon)] = i
        for i, mon in enumerate(list_of_day):
            X[np.where(X == mon)] = i

        X = np.array(X, dtype = np.float64)
        y = np.array(y, dtype = np.float64)

    if name == 'synthetic1':
        X, y = LoadData(url = "https://raw.githubusercontent.com/ttgadaev/SampleSize/master/datasets/synthetic1.csv", 
                        path = folder_path + 'synthetic1.csv')

        X = np.array(X, dtype = np.float64)
        y = np.array(y, dtype = np.float64)

    if name == 'synthetic2':
        X, y = LoadData(url = "https://raw.githubusercontent.com/ttgadaev/SampleSize/master/datasets/synthetic2.csv", 
                        path = folder_path + 'synthetic2.csv')

        X = np.array(X, dtype = np.float64)
        y = np.array(y, dtype = np.int64)

    return X, y
    
def select_features(X, y):
    if len(list(set(list(y)))) == 2:
        model = Logit(y, X)
    else:
        model = OLS(y, X)
    res = model.fit(disp = False)
    features = ind = multitest.multipletests(res.pvalues, method='holm')[0]	
    X = X[:, features]
    return X

def preprocess(X,y):
    X = scale(X)
    X = sm.add_constant(X)
    X = select_features(X, y)

    if len(list(set(list(y)))) != 2:
        w_hat = get_params(X, y) 
        y_hat = X@w_hat
        y = (y - (y - y_hat).mean())/np.std(y - y_hat)
        pass

    return X, y




