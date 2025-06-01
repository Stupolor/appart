import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_absolute_percentage_error
import seaborn as sns
import pickle



data = pd.read_csv('data/processed/nnapartment_more_info.csv')
pd.set_option('display.width', None)

data.info()

X = data.drop(['price'], axis=1)
y = data['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=12, stratify= data.rooms_count)

obj_cols = list(X_train.columns[X_train.dtypes=='object'])
num_cols = list(X_train.columns[X_train.dtypes!='object'])
print(obj_cols, num_cols, sep='\n')

ct = ColumnTransformer(
    [
        ('categorical', OneHotEncoder(handle_unknown='ignore'), obj_cols),
        ('numerical', StandardScaler(), num_cols)
    ],
    sparse_threshold=0
)

pipe = Pipeline(
    [
        ('data_tranfsormer', ct),
        ('feature_construction', PolynomialFeatures(degree=2, include_bias=False)),
        ('feature_selection', SelectFromModel(estimator=Lasso(10.0, max_iter=10000))),
        ('ridge', Ridge(alpha=1.0))
    ]
)

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
mean_absolute_percentage_error = mean_absolute_percentage_error(y_test, y_pred)
print(mean_absolute_percentage_error)

sns.scatterplot(x=y_test, y=y_pred)
sns.lineplot(x=y_test, y=y_test)
plt.show()

model_pkl_file = 'data/model/apartment_prices_regression.pkl'
with open(model_pkl_file, 'wb') as file:
    pickle.dump(pipe, file)