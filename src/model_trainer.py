import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
import optuna
from dataset import Dataset
from typing import Iterator, Literal, Sequence
import joblib
from typing import Any
from scipy.stats import pearsonr


class ModelTrainer():
    def __init__(self,
        n_outer_folds: int=3,
        n_inner_folds: int=5,
        cv_type: Literal['kfold', 'groupkfold']='kfold',
        shuffle: bool=True,
        random_state: int | np.random.RandomState | None=None) -> None:
        if cv_type == 'kfold':
            self._outer_cv = KFold(n_splits=n_outer_folds, shuffle=shuffle, random_state=random_state)
            self._inner_cv = KFold(n_splits=n_inner_folds, shuffle=shuffle, random_state=random_state)
        else: 
            self._outer_cv = GroupKFold(n_splits=n_outer_folds, shuffle=shuffle, random_state=random_state) # type: ignore
            self._inner_cv = GroupKFold(n_splits=n_inner_folds, shuffle=shuffle, random_state=random_state) # type: ignore
        self._random_state = random_state

    def split(self, ds: Dataset) -> Iterator:
        X, Y = ds()
        return self._outer_cv.split(X, Y, X['Protein'].to_numpy())

    def hyperparameter_search(self, ds: Dataset,
                          n_trails: int|None=None,
                          timeout: int|None=None,
                          n_jobs: int=1,
                          study_name: str|None=None) -> list[MultiOutputRegressor]:
        X, Y = ds()
        outer_models = []
            
        for outer_fold, (train_idx, test_idx) in enumerate(self._outer_cv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            Y_train, Y_test = Y.iloc[train_idx], Y.iloc[test_idx]

            iteration_counts = []

            def objective(trial: optuna.trial.Trial) -> float | Sequence[float]:
                params = {
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 10, 50),
                    "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
                    "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
                }

                # Use early stopping in inner CV
                model = MultiOutputRegressor(
                    HistGradientBoostingRegressor(
                        **params,
                        early_stopping=True,
                        validation_fraction=.1,
                        random_state=self._random_state,
                        categorical_features=ds.categorical_columns
                    )
                )

                scores = []
                for inner_train_idx, val_idx in self._inner_cv.split(X_train, groups=X_train['Protein'].to_numpy()):
                    X_inner_train, X_val = X_train.iloc[inner_train_idx], X_train.iloc[val_idx]
                    Y_inner_train, Y_val = Y_train.iloc[inner_train_idx], Y_train.iloc[val_idx]
                    model.fit(X_inner_train, Y_inner_train)

                    preds = model.predict(X_val)
                    score = root_mean_squared_error(Y_val, preds)
                    scores.append(score)

                    # Save iteration counts
                    for est in model.estimators_:
                        if isinstance(est, HistGradientBoostingRegressor):
                            iteration_counts.append(est.n_iter_)

                return float(np.mean(scores))

            seed = self._random_state if type(self._random_state) == int else None
            sampler = optuna.samplers.TPESampler(seed=seed)
            study_name_ = None if study_name is None else f'{study_name}_{outer_fold}'
            study = optuna.create_study(direction="minimize", study_name=study_name_, sampler=sampler)
            study.optimize(objective, n_trials=n_trails, timeout=timeout, n_jobs=n_jobs)

            best_params = study.best_params

            # Dynamically determine max_iter
            mean_iter = np.mean(iteration_counts)
            std_iter = np.std(iteration_counts)
            max_iter_final = int(mean_iter + std_iter)

            print(f"[Fold {outer_fold+1}] Mean Iter: {mean_iter:.2f}, Std: {std_iter:.2f} → Using max_iter={max_iter_final}")

            # Train final model on full outer training set (no early stopping)
            final_model = MultiOutputRegressor(
                HistGradientBoostingRegressor(
                    **best_params,
                    max_iter=max_iter_final,
                    early_stopping=False,
                    random_state=self._random_state,
                    categorical_features=ds.categorical_columns
                )
            )
            final_model.fit(X_train, Y_train)

            preds = final_model.predict(X_test)
            outer_models.append(final_model)

        return outer_models

def save_model(model: Any, fname: str) -> None:
    joblib.dump(model, fname)

def load_model(fname: str) -> Any:
    return joblib.load(fname)

def compute_metrics(y_true, y_pred) -> pd.DataFrame:
    index = ['RMSE', 'MAE', 'R2', 'Pearson correlation']
    columns = ['Diameter (µm)', 'strain (mm/mm)', 'strength (MPa)', 'Youngs Modulus (Gpa)', 'Toughness (MJ m-3)']
    data = [root_mean_squared_error(y_true, y_pred, multioutput='raw_values'),
            mean_absolute_error(y_true, y_pred, multioutput='raw_values'),
            r2_score(y_true, y_pred, multioutput='raw_values'),
            pearsonr(y_true, y_pred).statistic] # type: ignore
    return pd.DataFrame(data, index, columns)