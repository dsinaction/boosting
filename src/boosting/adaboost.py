import sklearn
import numpy as np


class AdaBoostClassifier:
    '''An AdaBoost classifier.'''
    
    def __init__(self, base_estimator, n_estimators):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimators = []
        self.estimators_weights = []
        self.estimators_errors = []
        
    def fit(self, X, y):
        '''Build a boosted classifier.'''
        weights = np.full(X.shape[0], 1/X.shape[0])
        for i in range(self.n_estimators):
            weights = self._fit_estimator(X, y, weights)
            
    def _fit_estimator(self, X, y, weights):
        clf = sklearn.base.clone(self.base_estimator)
        clf.fit(X, y, sample_weight=weights)
        yhat = clf.predict(X)

        error = weights[y != yhat].sum()
        estimator_weight = 0.5*np.log((1 - error)/error)

        weights = weights*np.exp(-yhat*y*estimator_weight)
        weights = weights/weights.sum()

        self.estimators.append(clf)
        self.estimators_weights.append(estimator_weight)
        self.estimators_errors.append(error)
        
        return weights
    
    def predict(self, X):
        '''Predict classes.'''
        preds = [ 
            estimator_weight*estimator.predict(X) 
            for estimator, estimator_weight in zip(self.estimators, 
                                                   self.estimators_weights) 
        ]
        preds = np.stack(preds, axis=1)
        return np.array([1 if x > 0 else -1 for x in preds.sum(axis=1)])