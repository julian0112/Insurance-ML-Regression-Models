import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder, QuantileTransformer
from scipy.stats import shapiro, zscore
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('insurance.csv')

le = LabelEncoder()
le.fit(data.sex.drop_duplicates()) 
data.sex = le.transform(data.sex)
# smoker or not
le.fit(data.smoker.drop_duplicates()) 
data.smoker = le.transform(data.smoker)
#region
le.fit(data.region.drop_duplicates()) 
data.region = le.transform(data.region)

data['charges']= np.log(data['charges'])

X = data.drop('charges',axis=1) # Independet variable
y = data['charges'] # dependent variable


ss = StandardScaler()
kf = KFold(shuffle=True, random_state=23, n_splits=4)

alphas = np.geomspace(1e-05, 10, 7)
alphas2 = [0.005, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 80]
l1_ratios = np.linspace(0.1, 0.9, 9)

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------

pipe_lr = Pipeline([('pf', PolynomialFeatures(degree=3)), ('ss', StandardScaler()), ('lr', LinearRegression())])
lr_predictions = cross_val_predict(pipe_lr, X, y, cv=kf)

print('R^2 LINEAR REGRESSION:', r2_score(y, lr_predictions))
print('MSE LINEAR REGRESSION:', np.sqrt(mean_squared_error(y, lr_predictions)))

pipe_lasso = Pipeline([('pf', PolynomialFeatures(degree=6)), ('ss', StandardScaler()), ('lasso', Lasso(alpha=0.005))])
lasso_predictions = cross_val_predict(pipe_lasso, X, y, cv=kf)

print('R^2 LASSO:', r2_score(y, lasso_predictions))
print('MSE LASSO REGRESSION:',  np.sqrt(mean_squared_error(y, lasso_predictions)))

pipe_ridge = Pipeline([('pf', PolynomialFeatures(degree=2)), ('ss', StandardScaler()), ('ridge', Ridge(alpha=1))])
ridge_predictions = cross_val_predict(pipe_ridge, X, y, cv=kf)

print('R^2 RIDGE:', r2_score(y, ridge_predictions))
print('MSE RIDGE REGRESSION:',  np.sqrt(mean_squared_error(y, ridge_predictions)))

pipe_elasticNet = Pipeline([('pf', PolynomialFeatures(degree=2)), ('ss', StandardScaler()), ('elasticNet', ElasticNet(alpha=0.005, l1_ratio=0.1))])
elasticNet_predictions = cross_val_predict(pipe_elasticNet, X, y, cv=kf)

print('R^2 ELASTIC NET:', r2_score(y, elasticNet_predictions))
print('MSE ELASTIC NET REGRESSION:',  np.sqrt(mean_squared_error(y, elasticNet_predictions)))


# estimator = Pipeline([("polynomial_features", PolynomialFeatures()),
#         ("scaler", StandardScaler()),
#         ("elasticNet", ElasticNet(max_iter=10000))])

# params = {
#     'polynomial_features__degree': [1, 2, 3, 4, 5, 6],
#     'elasticNet__alpha': alphas2,
#     'elasticNet__l1_ratio': l1_ratios
# }

# grid = GridSearchCV(estimator, params, cv=kf)
# grid.fit(X, y)
# print(grid.best_score_, grid.best_params_, mean_squared_error(y, grid.predict(X)))

# plt.figure()
# sns.heatmap(data.corr(), annot=True).set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
# sns.violinplot(x='smoker', y='charges', data=data, hue='sex', split=True)
# plt.show()

# plt.figure(figsize=(12,4))

# ax=plt.subplot(121)
# sns.histplot(data['charges'],bins=50,color='r',ax=ax, kde=True)
# ax.set_title('Distribution of insurance charges')

# ax=plt.subplot(122)
# sns.histplot(np.log10(data['charges']),bins=40,color='b',ax=ax, kde=True)
# ax.set_title('Distribution of insurance charges in $log$ sacle')
# ax.set_xscale('log')

# plt.show()

# df_ages = data.copy()
# age_ranges = range(18, 74, 10)
# age_labels = [f'{i}-{i+9}' for i in age_ranges[:-1]]
# df_ages['age_ranges'] = pd.cut(df_ages['age'], bins=age_ranges, labels=age_labels, right=False)
# df_ages['smoker'] = df_ages['smoker'].map({1: 'Yes', 0: 'No'})

# my_pal = {smoker: "#c03434" if smoker ==
#           "Yes" else "#0b5394" for smoker in df_ages["smoker"].unique()}

# plt.figure()
# sns.violinplot(df_ages, x='age_ranges', y='charges', hue='smoker', split=True, palette= my_pal, hue_order=['No', 'Yes']).set_title('Age Ranges vs. Charges by Smoking Habits')
# plt.show()

