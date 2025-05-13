import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, LabelEncoder, PowerTransformer
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split
from scipy.stats.mstats import winsorize
from scipy.stats import shapiro
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

def grid_search():
    alphas = [0.005, 0.05, 0.1, 0.3, 1, 3, 5, 10, 15]
    l1_ratios = np.linspace(0.1, 0.9, 9)
    
    X = data.drop('charges', axis=1)
    y = data['charges']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    estimator = Pipeline([("polynomial_features", PolynomialFeatures()),
        ("scaler", StandardScaler()),
        ("lasso", Lasso(max_iter=1000000))
        ])

    params = {
        'polynomial_features__degree': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'lasso__alpha': alphas,
    }

    grid = GridSearchCV(estimator, params, cv=4, scoring='neg_mean_squared_error')
    grid.fit(X_train, y_train)
    print("Mejores parámetros:", grid.best_params_)


# -------------------------------------------------------------------------------------------------------------------------------------------------------------------

X = data.drop('charges', axis=1)
y = data['charges']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

y_train = winsorize(y_train, limits=[0.05, 0.05])
y_winsorize = winsorize(y, limits=[0.05, 0.05])

pt = PowerTransformer(method='yeo-johnson')
y_train_transformed = pt.fit_transform(y_train.reshape(-1, 1)).flatten()
y_transformed = pt.fit_transform(y_winsorize.reshape(-1, 1)).flatten()

# ----------------------------------------- LINEAR TRAINING -----------------------------------------

feature_pipe_lr = Pipeline([
    ('scaler', StandardScaler())
])

model_lr = TransformedTargetRegressor(
    regressor=Pipeline([
        ('features', feature_pipe_lr),
        ('lr', LinearRegression())
    ]),
    transformer=pt
)

# Training Linear
model_lr.fit(X_train, y_train) 

predictions = model_lr.predict(X_test)

r2 = r2_score(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f'R² Linear Regression: {r2:.3f}')
print(f'RMSE Linear Regression: {rmse:.2f} USD\n')

# ----------------------------------------- RIDGE TRAINING -----------------------------------------

feature_pipe_ridge = Pipeline([
    ('poly', PolynomialFeatures(degree=2)),
    ('scaler', StandardScaler())
])

model_ridge = TransformedTargetRegressor(
    regressor=Pipeline([
        ('features', feature_pipe_ridge),
        ('ridge', Ridge(alpha=3))
    ]),
    transformer=pt
)

# Training Ridge
model_ridge.fit(X_train, y_train) 

predictions = model_ridge.predict(X_test)

r2 = r2_score(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f'R² Ridge Regression: {r2:.3f}')
print(f'RMSE Ridge Regression: {rmse:.2f} USD\n')

# ----------------------------------------- LASSO TRAINING -----------------------------------------

feature_pipe_lasso = Pipeline([
    ('poly', PolynomialFeatures(degree=6)),
    ('scaler', StandardScaler())
])

model_lasso = TransformedTargetRegressor(
    regressor=Pipeline([
        ('features', feature_pipe_lasso),
        ('lasso', Lasso(alpha=0.005))
    ]),
    transformer=pt
)

# Training Lasso
model_lasso.fit(X_train, y_train) 

predictions = model_lasso.predict(X_test)

r2 = r2_score(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f'R² Lasso Regression: {r2:.3f}')
print(f'RMSE Lasso Regression: {rmse:.2f} USD\n')

# ----------------------------------------- ELASTIC NET TRAINING -----------------------------------------

feature_pipe_elastic = Pipeline([
    ('poly', PolynomialFeatures(degree=6)),
    ('scaler', StandardScaler())
])

model_elastic = TransformedTargetRegressor(
    regressor=Pipeline([
        ('features', feature_pipe_elastic),
        ('elastic', ElasticNet(alpha=0.005, l1_ratio=0.1, max_iter=100000))
    ]),
    transformer=pt
)

# Training Elastic
model_elastic.fit(X_train, y_train) 

predictions = model_elastic.predict(X_test)

r2 = r2_score(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f'R² Elastic Net Regression: {r2:.3f}')
print(f'RMSE Elastic Net Regression: {rmse:.2f} USD\n')


def correlation_heatmap():
    plt.figure()
    sns.heatmap(data.corr(), annot=True, cmap='crest').set_title('Correlation Heatmap', fontdict={'fontsize':14}, pad=12)
    plt.show()

def charges_distribution(y_transformed, y):
      
    plt.figure(figsize=(12,4))

    ax=plt.subplot(121)
    sns.histplot(y,bins=50,color='#57a490',ax=ax, kde=True)
    ax.set_title('Distribution of Insurance Charges', fontsize=16)
    ax.set_xlabel('Charges', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)

    ax=plt.subplot(122)
    sns.histplot(y_transformed,bins=50,color='#2c3172',ax=ax, kde=True)
    ax.set_title('Distribution of Insurance Charges with Yeo-Johnson', fontsize=16)
    ax.set_xlabel('Charges', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)

    plt.show()

def violin_smoker_age_charges():
    df_ages = data.copy()
    age_ranges = range(18, 74, 10)
    age_labels = [f'{i}-{i+9}' for i in age_ranges[:-1]]
    df_ages['age_ranges'] = pd.cut(df_ages['age'], bins=age_ranges, labels=age_labels, right=False)
    df_ages['smoker'] = df_ages['smoker'].map({1: 'Yes', 0: 'No'})

    my_pal = {smoker: "#57a490" if smoker ==
              "Yes" else "#2c3172" for smoker in df_ages["smoker"].unique()}

    plt.figure()
    ax = sns.violinplot(df_ages, x='age_ranges', y='charges', hue='smoker', split=True, palette= my_pal, hue_order=['No', 'Yes']).set_title('Age Ranges vs. Charges by Smoking Habits', fontsize=16)
    plt.xlabel('Age Ranges', fontsize=14)
    plt.ylabel('Charges', fontsize=14)
    plt.show()

def prediction_vs_real():
    plt.scatter(y_test, predictions, alpha=0.5, c='#25788c', cmap='crest',)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Real Values (USD)')
    plt.ylabel('Predictions (USD)')
    plt.title('Ridge Regression: Predictions vs. Real Values')
    plt.show()

def linear_vs_ridge():
    results = pd.DataFrame({
    'Model': ['Linear Regression', 'Ridge Regression'],
    'R2': [0.685, 0.863],
    'RMSE': [6994.30, 4605.22]
    })
        
    plt.figure(figsize=(12, 7))

    ax=plt.subplot(121)
    ax1= sns.barplot(data=results, x='Model', y='R2', palette='crest', hue='Model', legend=False)
    ax.set_title('Linear and Ridge Model R² Score', fontsize=14)
    ax.set_ylabel('R² Score', fontsize=14)
    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylim(0, 1)
    for p in ax1.patches:
        for i, v in enumerate(results['R2']):
            ax1.text(i, v+0.02, f'{v:.3f}', ha='center', fontsize=14)

    ax=plt.subplot(122)
    ax2= sns.barplot(data=results, x='Model', y='RMSE', palette='crest', hue='Model', legend=False)
    ax.set_title('Linear and Ridge Models RMSE', fontsize=14)
    ax.set_ylabel('RMSE (USD)', fontsize=14)
    ax.set_xlabel('Model', fontsize=14)
    for p in ax2.patches:
        for i, v in enumerate(results['RMSE']):
            ax2.text(i, v+200, f'${v:.2f}', ha='center', fontsize=14)

    plt.show()
