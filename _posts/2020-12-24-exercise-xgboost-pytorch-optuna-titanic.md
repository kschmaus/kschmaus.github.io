---
layout: post
title:  "Exercise: Using XGBoost, Pytorch, Optuna for Kaggle Titanic Modeling"
date:   2020-12-24 00:00:00 -0000
categories: blog
---


After working at [Stitch Fix](https://multithreaded.stitchfix.com/algorithms/) for 6 years, I've taken just under 3 months off from any data science work. Now I'm getting ready to head back on the job market agan, and I decided to start working on some "exercises" to get back in the swing of things. In this first exercise (of many?), I fit some models on the [Kaggle Titanic Survival dataset](https://www.kaggle.com/c/titanic), a pretty simple and small "hello world" supervised learning problem. I fit models using [PyTorch](https://pytorch.org/) and [XGBoost](https://xgboost.readthedocs.io/en/latest/), and the hyperparameter optimization framework [Optuna](https://optuna.org/).

I am not an expert in any of these tools, hence why I'm calling this an exercise 😉. If you notice any mistakes I've made or have any suggestions, please don't hesitate to let me know.


```python
import typing

import numpy as np
import optuna
import pandas as pd
import plotnine as pn
import skorch
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from xgboost.sklearn import XGBClassifier

DEVICE = torch.device("cpu")
NUM_WORKERS = 4
SEED = 24601

torch.manual_seed(SEED)
```

# Section 1: Processing Data

Kaggle practioners are provided with a training dataset with features and a response, and a testing dataset with only features. Their task is to to predict a passenger's chance of surviving the [Titanic](https://en.wikipedia.org/wiki/Sinking_of_the_Titanic). In this exercise, I'm not going to spend much time doing "exploratory data analysis", or explaining the relatively simple feature processing I conduct below. However, I will give a brief description of the fields in our dataset.

- `PassengerId`: Passenger's unique identifier.
- `Survived`: Binary response, did the passenger survive the sinking of the Titanic.
- `Pclass`: Fare class of the passenger's ticket.
- `Name`: Passenger's name. (I drop this field for simplicity)
- `Sex`: Passenger's sex.
- `Age`: Passenger's age.
- `SibSp`: Number or siblings or spouses a passenger has aboard the titanic.
- `Parch`: Number of parents or children a passenger has aboard the titanic.
- `Ticket`: Passenger's ticket number.
- `Fare`: Passenger's ticket price.
- `Cabin`: Passenger's cabin number.
- `Embarked`: Passenger's port of embarkation.

Note, for simplicity, I drop the `Name` field. There is some interesting information embedded within this field, but I wasn't interested in processing it for this exercise. Similarly, for simplicities sake, I opt for relatively simple missing value imputation strategies.


```python
df_train = pd.read_csv("../data/train.csv")
df_test = pd.read_csv("../data/test.csv")
df = pd.concat([df_train, df_test])
df.head()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1.0</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1.0</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1.0</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0.0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>


```python
# utility functions
def first_letter(x) -> typing.List:
    if type(x) == list:
        return list(set([e[0] for e in x]))
    else:
        return []


def cabin_average(x) -> float:
    x_int = [int(e[1:]) for e in x if len(e) > 1]
    if len(x_int) > 0:
        return float(np.mean(x_int))
    else:
        return 0
```


```python
class FeaturePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.mlb_cabin_letter = MultiLabelBinarizer()
        self.ohe = OneHotEncoder(handle_unknown="ignore")
        self.ohe_columns = ["Embarked", "Pclass", "Sex", "SibSp", "Parch"]
        self.untransformed_columns = ["Age", "Fare"]
        self.cabin_na_fill = "Z"
        self.duplicated_tickets_ = None
        self.median_age_ = None
        self.median_fare_ = None
        self.embarked_mode_ = None

    def fit(self, X: pd.DataFrame):
        self.median_age_ = X.Age.median()
        self.median_fare_ = X.Fare.median()
        self.embarked_mode_ = X.Embarked.mode()[0]

        cabin_letter = (
            X.Cabin.fillna(self.cabin_na_fill).str.split(" ").apply(first_letter)
        )
        self.mlb_cabin_letter.fit(cabin_letter)

        self.ohe.fit(X[self.ohe_columns].dropna())

        self.duplicated_tickets_ = X.Ticket[X.Ticket.duplicated()].values

        return self

    def transform(self, X: pd.DataFrame):
        X.Age = X.Age.fillna(self.median_age_)
        X.Fare = X.Fare.fillna(self.median_fare_)
        X.Embarked = X.Embarked.fillna(self.embarked_mode_)

        cabin = X.Cabin.fillna(self.cabin_na_fill)

        x_n_cabins = cabin.str.strip().str.split(" ").apply(len).values.reshape((-1, 1))

        cabin_letter = cabin.str.split(" ").apply(first_letter)
        x_cabin_letter = self.mlb_cabin_letter.transform(cabin_letter)
        x_cabin_n = cabin_letter.apply(len).values.reshape((-1, 1))
        x_cabin_average = (
            cabin.str.split(" ").apply(cabin_average).values.reshape((-1, 1))
        )

        x_ticket_group = (
            X.Ticket.isin(self.duplicated_tickets_).astype(int).values.reshape((-1, 1))
        )

        x_ohe = self.ohe.transform(X[self.ohe_columns]).toarray()

        x_untransformed = X[self.untransformed_columns]

        x = np.concatenate(
            [
                x_n_cabins,
                x_cabin_letter,
                x_cabin_n,
                x_cabin_average,
                x_ticket_group,
                x_ohe,
                x_untransformed,
            ],
            axis=1,
        )
        return x
```


```python
feature_preprocessor = FeaturePreprocessor()
feature_preprocessor.fit(df)
x_train = feature_preprocessor.transform(df_train)
x_test = feature_preprocessor.transform(df_test)
y_train = df_train.Survived.values

x_train.shape  # (891, 38)
```


# Section 2: Fitting XGBoost Model

In this section we define a function, `cv_xgb_nll`, that takes in XGB classifier initialization parameters, and then fits K XGBoost classifier models using stratified cross-validation. `cv_xgb_nll` then generates predictions for each slice of out-of-sample data, and returns mean out-of-sample  negative log loss. Next, I use the Optuna framework to tune model parameters. Finally, we fit one last XGBoost classifier on the full training data using the best fitting parameters.

If we had a large amount of observations or features, it might not be feasible to fit 10 versions of our model for every set of parameters. However, as our training data has under 1000 observations (and not too many features), the computational is negligible. 

I'm only using some of the many XGBoost parameters, those that made sense to me on an initial reading of the [paramters page](https://xgboost.readthedocs.io/en/latest/parameter.html). 



```python
def cv_xgb_nll(
    x: np.ndarray,
    y: np.ndarray,
    colsample_bytree: float,
    gamma: float,
    learning_rate: float,
    max_depth: int,
    n_estimators: int,
    reg_alpha: float,
    reg_lambda: float,
    subsample: float,
) -> float:
    test_log_loss = []
    stratified_k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=24601)
    for train_index, test_index in stratified_k_fold.split(x, y):
        x_cv_train = x[train_index]
        x_cv_test = x[test_index]
        y_cv_train = y[train_index]
        y_cv_test = y[test_index]

        xgb_classifier = XGBClassifier(
            colsample_bytree=colsample_bytree,
            eval_metric="logloss",
            gamma=gamma,
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=SEED,
            subsample=subsample,
            use_label_encoder=False,
        )
        xgb_classifier.fit(x_cv_train, y_cv_train)
        prob_cv_test = xgb_classifier.predict_proba(x_cv_test)[:, 1].astype(np.float64)
        test_log_loss.append(log_loss(y_true=y_cv_test, y_pred=prob_cv_test))

    return float(np.mean(test_log_loss))

```


```python
def cv_xgb_optuna_objective(trial: optuna.Trial) -> float:
    return cv_xgb_nll(
        x=x_train,  # from global namespace
        y=y_train,  # from global namespace
        colsample_bytree=trial.suggest_uniform("colsample_bytree", 0.1, 1.0),
        gamma=trial.suggest_loguniform("gamma", 1e-8, 1.0),
        learning_rate=trial.suggest_loguniform("learning_rate", 1e-8, 1.0),
        max_depth=trial.suggest_int("max_depth", 1, 30),
        n_estimators=trial.suggest_int("n_estimators", 50, 500),
        reg_alpha=trial.suggest_loguniform("reg_alpha", 1e-8, 1.0),
        reg_lambda=trial.suggest_loguniform("reg_lambda", 1e-8, 1.0),
        subsample=trial.suggest_uniform("subsample", 1e-1, 1.0),
    )

```


```python
xgb_sampler = optuna.samplers.TPESampler(seed=SEED)
xgb_study = optuna.create_study(study_name="cv-xgb-titanic", sampler=xgb_sampler)
xgb_study.optimize(cv_xgb_optuna_objective, n_trials=100)
pd.DataFrame([xgb_study.best_params]
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>colsample_bytree</th>
      <th>gamma</th>
      <th>learning_rate</th>
      <th>max_depth</th>
      <th>n_estimators</th>
      <th>reg_alpha</th>
      <th>reg_lambda</th>
      <th>subsample</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.389661</td>
      <td>0.254362</td>
      <td>0.01056</td>
      <td>21</td>
      <td>455</td>
      <td>1.301069e-07</td>
      <td>0.604678</td>
      <td>0.861531</td>
    </tr>
  </tbody>
</table>
</div>


```python
xgb_study_df = xgb_study.trials_dataframe()
xgb_total_minutes = xgb_study_df.duration.sum().seconds / 60
(
    pn.ggplot(xgb_study_df, pn.aes(x="number", y="value"))
    + pn.geom_line()
    + pn.theme_bw()
    + pn.xlab("# of trials")
    + pn.ylab("Mean CV Log Loss ")
    + pn.ggtitle(f"Optuna Objective\n{xgb_total_minutes:.1f} Minutes")
)
```


![png](/assets/2020-12-24-exercise-xgboost-pytorch-optuna-titanic/output_12_0.png)


```python
xgb_classifier = XGBClassifier(
    eval_metric="logloss",
    random_state=SEED,
    use_label_encoder=False,
    **xgb_study.best_params,
)
xgb_classifier.fit(x_train, y_train)
xgb_pred = xgb_classifier.predict_proba(x_test)[:, 1]
```

# Section 3: Fitting PyTorch Model

My PyTorch approach is similar to my XGBoost approach. I'm using the [skorch](https://github.com/skorch-dev/skorch) framework, which gives a scikit-learn API for PyTorch models. K-fold cross validation feels slightly more cumbersome (and less common?) with a PyTorch model, but again, because the size of our dataset is small, I'm happy with the approach. Initially I wanted to enable pruning (early stoppage during tuning) using optuna, but this proved difficult with the k-fold setup.


```python
def pt_model(
    in_features: int,
    hidden_layers: typing.List[int],
    dropout_rates: typing.List[float],
    epochs: int,
    lr: float,
    weight_decay: float,
) -> skorch.NeuralNetBinaryClassifier:
    assert len(hidden_layers) == len(dropout_rates)

    layers = []

    for hidden_layer, dropout_rate in zip(hidden_layers, dropout_rates):
        layers.append(nn.Linear(in_features, hidden_layer))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        in_features = hidden_layer

    layers.append(nn.Linear(in_features, 1))
    layers.append(nn.Sigmoid())

    return skorch.NeuralNetBinaryClassifier(
        module=nn.Sequential(*layers),
        max_epochs=epochs,
        optimizer=optim.AdamW,
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        iterator_train__shuffle=True,
        train_split=None,
        criterion=torch.nn.BCELoss,
    )
```


```python
def cv_pt_nll(
    x: np.ndarray,
    y: np.ndarray,
    hidden_layers: typing.List[int],
    dropout_rates: typing.List[float],
    epochs: int,
    lr: float,
    weight_decay: float,
):
    test_log_loss = []
    stratified_k_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=24601)
    for train_index, test_index in stratified_k_fold.split(x, y):
        x_cv_train = x[train_index].astype(np.float32)
        x_cv_test = x[test_index].astype(np.float32)
        y_cv_train = y[train_index].astype(np.float32)
        y_cv_test = y[test_index].astype(np.float32)

        pt_classifier = pt_model(
            in_features=x.shape[1],
            hidden_layers=hidden_layers,
            dropout_rates=dropout_rates,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
        )
        pt_classifier.fit(x_cv_train, y_cv_train)
        prob_cv_test = pt_classifier.predict_proba(x_cv_test).astype(np.float64)
        test_log_loss.append(log_loss(y_true=y_cv_test, y_pred=prob_cv_test))

    return float(np.mean(test_log_loss))
```


```python
def cv_pt_optuna_objective(trial: optuna.Trial) -> float:
    n_hidden_layers = trial.suggest_int("n_hidden_layers", 1, 5)
    hidden_layers = []
    dropout_rates = []
    for i in range(n_hidden_layers):
        hidden_layers.append(trial.suggest_int(f"hidden_layer{i}", 4, 256))
        dropout_rates.append(trial.suggest_uniform(f"dropout_rate{i}", 0.0, 0.5))

    return cv_pt_nll(
        x=x_train,
        y=y_train,
        hidden_layers=hidden_layers,
        dropout_rates=dropout_rates,
        epochs=trial.suggest_int("epochs", 10, 100),
        lr=trial.suggest_loguniform("lr", 1e-5, 1e-1),
        weight_decay=trial.suggest_loguniform("weight_decay", 1e-8, 1.0),
    )
```


```python
pt_sampler = optuna.samplers.TPESampler(seed=SEED)
pt_study = optuna.create_study(study_name="cv-xgb-titanic", sampler=pt_sampler)
pt_study.optimize(cv_pt_optuna_objective, n_trials=100)
pd.DataFrame([pt_study.best_params])
```


```python
pt_study_df = pt_study.trials_dataframe()
pt_total_minutes = pt_study_df.duration.sum().seconds / 60
(
    pn.ggplot(pt_study_df, pn.aes(x="number", y="value"))
    + pn.geom_line()
    + pn.theme_bw()
    + pn.ylim(0.4, 0.6)
    + pn.xlab("# of trials")
    + pn.ylab("Mean CV Log Loss ")
    + pn.ggtitle(f"PyTorch Objective\n{pt_total_minutes:.1f} Minutes")
)
```


![png](/assets/2020-12-24-exercise-xgboost-pytorch-optuna-titanic/output_20_0.png)


```python
pt_classifier = pt_model(
    in_features=x_train.shape[1],
    hidden_layers=[pt_study.best_params["hidden_layer0"]],
    dropout_rates=[pt_study.best_params["dropout_rate0"]],
    epochs=pt_study.best_params["epochs"],
    lr=pt_study.best_params["lr"],
    weight_decay=pt_study.best_params["weight_decay"],
)
pt_classifier.fit(x_train.astype(np.float32), y_train.astype(np.float32))
pt_pred = pt_classifier.predict_proba(x_test.astype(np.float32))
```

# Section 4: Summary

The XGBoost study appears to have a better fit than the PyTorch study. Their predictions agree directionally (for the most part), but there are several observations with marked differences. A possible extension of this exercise would be fitting some sort of voting-ensemble of these two models, with relative vote shared tuned using the previous hold-out folds. If you read this far, please share any thoughts or questions you might have. 

After submitting to the kaggle challenge, [I got an accuracy of 0.76794](https://www.kaggle.com/c/titanic/leaderboard). Let me know if you have an approach that scores higher.  


```python
xgb_study.best_value  # 0.4068
pt_study.best_value  # 0.4243
```




```python
pred_df = pd.DataFrame({"xgb_pred": xgb_pred, "pt_pred": pt_pred})
(
    pn.ggplot(pred_df, pn.aes(x="xgb_pred", y="pt_pred"))
    + pn.geom_point(alpha=0.3)
    + pn.theme_bw()
    + pn.geom_abline(intercept=0, slope=1, linetype="dashed")
    + pn.xlab("XGBoost Predictions")
    + pn.ylab("PyTorch Predictions")
)
```


![png](/assets/2020-12-24-exercise-xgboost-pytorch-optuna-titanic/output_25_0.png)

