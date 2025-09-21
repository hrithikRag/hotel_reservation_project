from scipy.stats import randint, uniform

LGBM_PARAMS={'n_estimators' : randint(50,500),
             'max_depth' : randint(2,50),
             'learning_rate' : uniform(0.01,0.1),
             'num_leaves' : randint(10,500),
             'boosting_type' :   ['gbdt','dart','goss']
}

RANDOM_SEARCH_PARAMS={
    'n_iter' : 5,
    'cv' : 2,
    'n_jobs' : -1,
    'verbose' : 2,
    'random_state' : 42,
    'scoring' : 'accuracy'
}