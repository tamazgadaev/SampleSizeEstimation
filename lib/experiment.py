from sklearn.preprocessing import scale
import statsmodels.api as sm
import m_models as mm
import numpy as np
import utils as ut
import os

from tqdm import tqdm as tqdm

dataset_name = 'boston'

X, y = ut.DataLoader(name=dataset_name)
X = scale(X)
y = scale(y)
X = sm.add_constant(X)

backup_folder = "backup/"
if not os.path.exists(backup_folder):
    os.mkdir(backup_folder)


datasets = []
sample_sizes = [28, 50, 100, 120, 140, 160, 180, 200, 220, 240]

name = dataset_name

data = {'name': name,
        'X': np.array(X),
        'y': np.array(y),
        'folder_path': backup_folder+name+"/",
        'backup': { 
                    'likelihood': [True, "likelihood.p"],
                    'cross_val': [True, "cross_val.p"],
                    'lagrange': [True, "lagrange.p"],
                    'likelihood_ratio': [True, "likelihood_ratio.p"],
                    'wald': [True, "wald.p"],
                    'apvc': [True, "apvc.p"],
                    'acc': [True, "acc.p"],
                    'alc': [True, "alc.p"],
                    'use': [True, "use.p"],
                   },
        'sample_sizes': np.array(sample_sizes),
        'methods': { 
                     'likelihood': lambda X, y, sample_sizes:  ut.get_error_track(X, y, sample_sizes=sample_sizes, k_for_meaning=1000, progress_bar=lambda x: tqdm(x, leave = False)),
                     'cross_val': lambda X, y: mm.cross_val(X, y, k_for_meaning=10),
                     'lagrange': lambda X, y: mm.lagrange(X, y),
                     'likelihood_ratio': lambda X, y: mm.likelihood_ratio(X, y),
                     'wald': lambda X, y: mm.wald(X, y, ind = 2),
                     'apvc': lambda X, y: mm.APVC(X, y, k_for_meaning=100, num = 5, progress_bar=lambda x: tqdm(x, leave = False), multiprocess = True),
                     'acc': lambda X, y: mm.ACC(X, y, k_for_meaning=10, maxiter=10, end = 700),
                     'alc': lambda X, y: mm.ALC(X, y, k_for_meaning=10, maxiter=10, end = 700),
                     'use': lambda X, y: mm.MAX_U(X, y, k_for_meaning=10, progress_bar=lambda x: tqdm(x, leave = False)),
                    },
}

datasets.append(data)


answers = ut.experiment_for_datasets(datasets, k_for_meaning = 50, test_train_proportional = 0.25, progress_bar=lambda x: tqdm(x, leave = False))


