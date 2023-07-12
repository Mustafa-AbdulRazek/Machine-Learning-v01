import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline


# create custom transformer that extracts cols passed as args to its costructor
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    # define fit method, nothing to do here
    def fit(self, X, y=None):
        return self

    # transformer method, to do what we need
    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        return X[self.feature_names]


# create custom transformer for categorical inputs
class CategoricalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, use_dates=['year', 'month', 'day']):
        self.use_dates = use_dates

    def fit(self, X, y=None):
        return self

    # helper function that extract year
    # @staticmethod
    def get_year(self, obj):
        return str(obj)[:4]

    # helper method that extract month
    # @staticmethod
    def get_month(self, obj):
        return str(obj)[4:6]

    # @staticmethod
    def get_day(self, obj):
        return str(obj)[6:8]

    # helper method that converts values to binary depending on input
    # @staticmethod
    def create_binary(self, obj):
        if obj == 0:
            return 'NO'
        else:
            return 'Yes'

    def transform(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        for val in self.use_dates:
            # exec("X.loc[:, '{}'] = X['date'].apply(self.get_{})".format(val, val))
            X.loc[:, val] = X['date'].apply(getattr(self, f"get_{val}"))
        print('transform cat transformer ')
        # drop unusable col
        X = X.drop('date', axis=1)

        # convert to binary for ohe
        X.loc[:, 'waterfront'] = X['waterfront'].apply(self.create_binary)
        X.loc[:, 'view'] = X['view'].apply(self.create_binary)
        X.loc[:, 'yr_renovated'] = X['yr_renovated'].apply(self.create_binary)
        print('success')
        return X.values


class NumericalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, bath_per_head=True, years_old=True):
        self.bath_per_head = bath_per_head
        self.years_old = years_old

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # check if needed
        if self.bath_per_head:
            # create new col
            X.loc[:, 'bath_per_bed'] = X['bathrooms']/X['bedrooms']
            # drop redundant col
            X.drop('bathrooms', axis=1)
        if self.years_old:
            # create new col
            X.loc[:, 'years_old'] = 2019 - X['yr_built']
            # drop redundant col
            X.drop('yr_built', axis=1)

        # convert infinity to Nan
        X = X.replace([np.inf, -np.inf], np.nan)

        return X.values


# define categorical inputs
categorical_inputs = ['date', 'waterfront', 'view', 'yr_renovated']

# define numerical inputs
numerical_inputs = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                    'condition', 'grade', 'sqft_basement', 'yr_built']

# define steps of categorical pipeline
categorical_pipeline = Pipeline(steps=[
    ('cat_selector', FeatureSelector(categorical_inputs)),
    ('cat_transformer', CategoricalTransformer()),
    ('ohe', OneHotEncoder(sparse=False))
])

# define steps of numerical pipeline
numerical_pipeline = Pipeline(steps=[
    ('num_selector', FeatureSelector(numerical_inputs)),
    ('num_transformer', NumericalTransformer()),
    ('imputer', SimpleImputer(strategy='median')),
    ('std_scaler', StandardScaler())
])

# combine transformers horizontally using FeatureUnion class
union_pipelines = FeatureUnion(transformer_list=[
    ('categorical_pipeline', categorical_pipeline),
    ('numerical_pipeline', numerical_pipeline)
])


# build a ML model and add to new pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
data = pd.read_csv(r'data\kc_house_data.csv')

X = data.drop('price', axis=1)
y = data['price'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# full pipeline here
full_pipeline = Pipeline(steps=[
    ('feature_union', union_pipelines),
    ('model', LinearRegression())
])

# fit
full_pipeline.fit(X_train, y_train)

# predict
y_pred = full_pipeline.predict(X_test)



