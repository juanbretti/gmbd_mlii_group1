# %%
import numpy as np
import pandas as pd
import sklearn as sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score,recall_score,precision_score, confusion_matrix
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

params = {'max_depth': None,
 'min_samples_leaf': 1,
 'min_samples_split': 2,
 'n_estimators': 250}
clf = RandomForestClassifier(**params)
clf.fit(X[reduced_features], y)

#Check accuracy of model after hyperparameter tuning 
print (f'Train Accuracy: {clf.score(X[reduced_features],y):.3f}')
print (f'Test Accuracy: {clf.score(X_val[reduced_features],y_val):.3f}')
# %%

params = {
    'n_estimators': [100, 250, 600],
    'max_depth': [None, 30, 100],
    'min_samples_split': [2, 10],
    'min_samples_leaf': [1, 2, 5] 
    }
cv_ = 3

rf_model_grid = RandomForestClassifier(random_state=0, n_jobs=-1)
rf_model_search = GridSearchCV(rf_model_grid, param_grid=params, scoring='accuracy', n_jobs=-1, cv=cv_, verbose=3)

rf_model_search.fit(X, y)

rf_model_after_search = rf_model_search.best_estimator_
rf_model_after_search_scores = cross_val_score(rf_model_after_search, X, y, scoring='accuracy', cv=3, n_jobs=-1)
print("Accuracy: %0.4f (+/- %0.2f)" % (np.median(rf_model_after_search_scores), np.std(rf_model_after_search_scores)))

# %%