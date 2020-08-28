import time
from tqdm import tqdm
import pickle
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import sklearn.metrics as smet


import numpy as np


class MlModel:
    def __init__(self, ml_model_name, param_grid):
        self.ml_model_name = ml_model_name
        self.param_grid = param_grid



    def train_all_parameters(self, X_train, Y_train, X_valid, Y_valid, save_metadata=False):
        metadata_list = []
        best_score = -np.inf
        for arg in tqdm([{k: v for k, v in zip(self.param_grid.keys(), comb)} for comb in
                         itertools.product(*[v for v in self.param_grid.values()])]):
            if self.ml_model_name=="lr":
                arg.update({"n_jobs": -1})
                ml_model_method = LogisticRegression(**arg)
            elif self.ml_model_name =="rf":
                arg.update({"n_jobs": -1})
                ml_model_method = RandomForestClassifier(**arg)
            elif self.ml_model_name == "mlp":
                ml_model_method = MLPClassifier(**arg)
            else:
                raise ValueError("model_name should be 'lr', 'rf' or 'mlp'")
            ts = time.time()
            ml_model_method.fit(X_train, Y_train)
            te = time.time()
            t_learning = te - ts

            ts = time.time()
            accuracy_train = ml_model_method.score(X_train, Y_train)
            f1_macro_score_train = smet.f1_score(ml_model_method.predict(X_train),Y_train, average='macro')
            balanced_accuracy_train = smet.balanced_accuracy_score(ml_model_method.predict(X_train),Y_train)
            accuracy_valid = ml_model_method.score(X_valid, Y_valid)
            f1_macro_score_valid = smet.f1_score(ml_model_method.predict(X_valid),Y_valid, average='macro')
            balanced_accuracy_valid = smet.balanced_accuracy_score(ml_model_method.predict(X_valid),Y_valid)
            te = time.time()
            t_predict = te - ts

            metadata = {"name": self.ml_model_name, "learning_time": t_learning, "predict_time": t_predict,
                        "accuracy_train": accuracy_train, "accuracy_valid": accuracy_valid, "parameter": arg,
                        "balanced_accuracy_valid":balanced_accuracy_valid, "f1_macro_score_valid":f1_macro_score_valid,
                        "balanced_accuracy_train":balanced_accuracy_train, "f1_macro_score_train":f1_macro_score_train}
            metadata_list.append(metadata)

            if accuracy_valid > best_score:
                best_score = accuracy_valid
                best_model = ml_model_method
                best_metadata = metadata
        print("Best model's parameters : %s" % str(best_metadata["parameter"]))

        if save_metadata:
            print(metadata_list)
            pickle.dump(metadata_list, open("data/metadata_%s.pkl" %self.ml_model_name ,"wb"))

        return best_model, best_metadata

