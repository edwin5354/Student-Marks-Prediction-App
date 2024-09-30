import pandas as pd
import numpy as np

df = pd.read_csv('./csv/processed.csv')

# Extract X & y variables 
X = df.drop(['Marks', 'Performance'], axis= 1)
y = df['Marks']

# Import train_test_split function
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Import svm model
from sklearn import svm
reg = svm.SVR(kernel='linear') # Linear Kernel
reg.fit(X_train, y_train)

# Predict the response for both train and test dataset
org_ytrain_pred = reg.predict(X_train)
org_ytest_pred = reg.predict(X_test)

# --------------------------------------------------------------------------------------------------------
# Try some hyperparameter tuning
from sklearn.model_selection import GridSearchCV 
from sklearn.svm import SVR

param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf','linear', 'poly', 'sigmoid']}

grid = GridSearchCV(SVR(), param_grid, refit = True, verbose = 3)

# fitting the model for grid search 
grid.fit(X_train, y_train) 

# print best parameter after tuning 
grid.best_params_ # {'C': 1000, 'gamma': 0.01, 'kernel': 'rbf'}
  
# print how our model looks after hyper-parameter tuning 
grid.best_estimator_ # SVR(C=1000, gamma=0.01)

# Create a tuned model for predictions
tuned_reg = svm.SVR(C = 1000, gamma = 0.01, kernel='rbf')
tuned_reg.fit(X_train, y_train)

# Predict the response for both train and test dataset after tunning
new_ytrain_pred = tuned_reg.predict(X_train)
new_ytest_pred = tuned_reg.predict(X_test)
# --------------------------------------------------------------------------------------------------------

# Evaluating the model (MAE, MSE, RMSE)
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error

def score(train_pred,test_pred,  is_optimized=False):
    train_dict = {
        'metrics': ['MAE', 'MSE', 'RMSE'],
        'Training': [
            np.round(mean_absolute_error(y_train, train_pred), 2),
            np.round(mean_squared_error(y_train, train_pred), 2),
            np.round(root_mean_squared_error(y_train, train_pred), 2)
        ],
        'Testing': [
            np.round(mean_absolute_error(y_test, test_pred), 2),
            np.round(mean_squared_error(y_test, test_pred), 2),
            np.round(root_mean_squared_error(y_test, test_pred), 2),
        ]
    }

    metrics_df = pd.DataFrame(train_dict)
    if not is_optimized:
        metrics_df.to_csv('./csv/org_svm_metrics.csv', index= False)
    else:
        metrics_df.to_csv('./csv/tuned_svm_metrics.csv', index= False)

# Original svm
score(org_ytrain_pred, org_ytest_pred,  is_optimized=False)

# Tuned svm
score(new_ytrain_pred, new_ytest_pred, is_optimized= True)
# --------------------------------------------------------------------------------------------------------

# save the model
import pickle
with open('./model/tuned_svm_model.pkl', 'wb') as tuned_svm_file: # save the svm model 
    pickle.dump(tuned_reg, tuned_svm_file)

with open('./model/scaler.pkl', 'wb') as scaler_file:
    pickle.dump(sc, scaler_file)
# --------------------------------------------------------------------------------------------------------
