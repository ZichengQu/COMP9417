import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

def grid_search(X, y, params):
    tuned_parameters = [{
                        "hidden_layer_sizes": [(16, 64, 256, 64, 16), (16, 64, 256, 64, 8)],
                        # "activation": ["identity", "logistic", "tanh", "relu"],
                        "solver": ["lbfgs", "sgd", "adam"],
                        # "alpha": np.arange(0.0001, 1, 0.01),
                        # "batch_size": np.arange(200, 5000, 100),
                        # "learning_rate": ["constant", "invscaling", "adaptive"],
                        # "learning_rate_init": np.arange(0.001, 1, 0.005),
                        # "power_t": ,
                        # "max_iter": np.arange(10, 500, 10),
                        # "shuffle": [True, False],
                        # "random_state": [],
                        # "tol": ,
                        # "momentum": np.arange(0.1, 1.01, 0.1),
                        # "nesterovs_momentum": [True, False],
                        # "early_stopping": [True, False]
                     }
                    ]
    classifier = MLPClassifier(**params)
    classifier = GridSearchCV(classifier ,tuned_parameters, verbose = 10)
    classifier.fit(X, y)
    print("The global best params", classifier.best_params_)
    return classifier