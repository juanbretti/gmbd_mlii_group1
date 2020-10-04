# %%
import numpy as np
import pandas as pd
import sklearn as sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score,recall_score,precision_score, confusion_matrix, classification_report
import xgboost as xgb

# %%
df = pd.read_csv('raw/csgo_round_snapshots.csv')
# %%
label_encoder = LabelEncoder()
df['bomb_planted']= label_encoder.fit_transform(df['bomb_planted'])
df['map']= label_encoder.fit_transform(df['map'])
# %%
num_cols = df.select_dtypes(include=['number']).columns
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols]) 
# %%
target= 'round_winner'
X, X_val, y, y_val= train_test_split(df.drop(target, axis=1),df[target],test_size=0.3, random_state=1,stratify=df[target])
# %%
reduced_features=['ct_players_alive', 't_grenade_flashbang', 't_weapon_ak47',
       'ct_weapon_sg553', 't_players_alive', 'ct_health', 't_weapon_sg553',
       't_grenade_molotovgrenade', 't_health', 'ct_weapon_awp', 't_weapon_awp',
       'ct_money', 't_money', 't_grenade_smokegrenade', 'ct_weapon_m4a4',
       't_armor', 'ct_armor', 't_helmets', 'bomb_planted']

# %%

import xgboost as xgb
from skopt import BayesSearchCV

params = {
        'learning_rate': [0.01, 0.3, 0.5],
        'min_child_weight': [None, 0, 1, 5, 10],
        'gamma': [None, 0, 0.5, 1, 1.5, 2, 5],
        'colsample_bytree': [None, 0.6, 0.8, 1.0],
        'max_depth': [10],
        'subsample': [0.75, 1],
        'n_estimators': [100, 500],
        'max_delta_step': [0.0],
        'colsample_bylevel': [1.0],
        'reg_alpha': [0.0],
        'reg_lambda': [1.0],
        'base_score': [0.5],
        'missing': [None]
        }

folds = 3
param_comb = 100
cv_ = 3

xgb_model_bayes = xgb.XGBClassifier(objective="binary:logistic", random_state=42, tree_method='gpu_hist', gpu_id=0, nthread=-1)
xgb_model_search_bayes = BayesSearchCV(xgb_model_bayes, search_spaces=params, n_iter=param_comb, scoring='accuracy', n_jobs=-1, cv=cv_, verbose=3, random_state=42)
xgb_model_search_bayes.fit(X, y)

# %%

xgb_model_after_bayes_search = xgb_model_search_bayes.best_estimator_
xgb_scores_bayes_tunned = cross_val_score(xgb_model_after_bayes_search, X_val, y_val, scoring='accuracy', cv=10)
print("Accuracy: %0.4f (+/- %0.2f)" % (np.median(xgb_scores_bayes_tunned), np.std(xgb_scores_bayes_tunned)))

# %%

# XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1.0,
#               colsample_bynode=1, colsample_bytree=0.8, gamma=0, gpu_id=0,
#               importance_type='gain', interaction_constraints='',
#               learning_rate=0.3, max_delta_step=0.0, max_depth=10,
#               min_child_weight=0, missing=None,
#               monotone_constraints='(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)',
#               n_estimators=480, n_jobs=-1, nthread=-1, num_parallel_tree=1,
#               random_state=42, reg_alpha=0.0, reg_lambda=1.0,
#               scale_pos_weight=1, subsample=0.9983775297693847,
#               tree_method='gpu_hist', validate_parameters=1, verbosity=None)

# Accuracy: 0.8905 (+/- 0.00)

# %%
model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1.0,
              colsample_bynode=1, colsample_bytree=0.8, gamma=0, gpu_id=0,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.3, max_delta_step=0.0, max_depth=10,
              min_child_weight=0, missing=None,
              monotone_constraints='(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)',
              n_estimators=480, n_jobs=-1, nthread=-1, num_parallel_tree=1,
              random_state=42, reg_alpha=0.0, reg_lambda=1.0,
              scale_pos_weight=1, subsample=0.9983775297693847,
              tree_method='gpu_hist', validate_parameters=1, verbosity=None)

model_trained = model.fit(X, y)

# %%

df_validation = pd.read_csv('raw/validation_set.csv')

label_encoder = LabelEncoder()
df_validation['bomb_planted']= label_encoder.fit_transform(df_validation['bomb_planted'])
df_validation['map']= label_encoder.fit_transform(df_validation['map'])

# %%
num_cols = df_validation.select_dtypes(include=['number']).columns
scaler = StandardScaler()
df_validation[num_cols] = scaler.fit_transform(df_validation[num_cols]) 

# %%
X_validation = df_validation.drop(['round_winner'], axis=1)
y_validation = df_validation['round_winner']

# %%
y_pred = model_trained.predict(X_validation)
accuracy_score(y_validation, y_pred)

# %%