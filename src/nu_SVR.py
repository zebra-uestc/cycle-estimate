from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, space_eval
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict
import numpy as np
from sklearn import svm

def optimize_nusvr(selected_X, selected_y, X, centers, total_cycle_baseline):
    # Define the hyperparameter search space
    space = {
        'C': hp.uniform('C', 0.0001, 10.0),
        'nu': hp.uniform('nu', 0.0001, 0.99),
        'degree': hp.choice('degree', [2, 3, 4, 5]),
        'max_iter': hp.choice('max_iter', [500, 2000, 5000, 10000]),
        'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
    }
    def objective(params):
        model = svm.NuSVR(C=params['C'], nu=params['nu'], max_iter=params['max_iter'], kernel=params['kernel'], degree=params['degree'])
        # Train the model using the entire training set
        model.fit(selected_X, selected_y)
        # Predict on the test set
        y_pred_test = model.predict(X)
        total_cycle_pred = np.sum(y_pred_test) + centers
        total_cycle_pred = int(total_cycle_pred)
        pred_error = np.abs(total_cycle_pred - total_cycle_baseline) / total_cycle_baseline * 100
        return pred_error
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=2000, trials=trials)
    # Retrieve the best hyperparameters
    best_params = space_eval(space, best)
    print("best params:", best_params)
    # Rebuild the model using the best hyperparameters
    best_model = svm.NuSVR(C=best_params['C'], nu=best_params['nu'], max_iter=best_params['max_iter'], kernel=best_params['kernel'], degree=best_params['degree'])
    best_model.fit(selected_X, selected_y)
    # Predict on the test set
    y_pred_test = best_model.predict(X)
    total_cycle_pred = np.sum(y_pred_test) + centers
    total_cycle_pred = int(total_cycle_pred)
    pred_error = np.abs(total_cycle_pred - total_cycle_baseline) / total_cycle_baseline * 100
    return pred_error