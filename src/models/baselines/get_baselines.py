import pyod
from pyod.models.ecod import ECOD
from pyod.models.pca import PCA
from pyod.models.kpca import KPCA
from pyod.models.ocsvm import OCSVM
from pyod.models.knn import KNN
from pyod.models.iforest import IForest
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.copod import COPOD
from pyod.models.gmm import GMM
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.hbos import HBOS


def get_baseline_model(cfg, model_name):
    n_components  = cfg.models.n_components
    num_jobs  = cfg.models.num_jobs
    num_jobs  = cfg.models.num_jobs
    random_state  = cfg.models.random_state
    contamination  = cfg.models.contamination
    use_ae  = cfg.models.use_ae
    epochs  = cfg.models.epochs

    if model_name =='PCA':
        return PCA(n_components=n_components)
    if model_name == 'MCD':
        return MCD(contamination=contamination)
    if model_name == 'OneClassSVM': 
        return OCSVM()
    if model_name == 'KNN': 
        return KNN(n_jobs=num_jobs)
    if model_name == 'LOF' : 
        return LOF(n_jobs=num_jobs)
    if model_name == 'IForest': 
        return IForest()
    if model_name == 'GMM':
        return GMM(n_components=n_components)
    if model_name == 'ECOD': 
        return ECOD(n_jobs=num_jobs)
    if model_name== 'KPCA':
        return KPCA()
    if model_name == 'COPOD': 
        return COPOD(n_jobs=num_jobs),
    if model_name == 'DeepSVDD': 
        return DeepSVDD(use_ae=use_ae, epochs=epochs, contamination=contamination,
                    random_state=random_state)
    
def get_all_baselines(cfg):
    baseline_model_dict = {}
    for model_name in cfg.models.model_List:
        baseline_model_dict[model_name] = get_baseline_model(cfg, model_name)
    return baseline_model_dict

