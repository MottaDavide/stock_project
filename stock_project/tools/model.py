from catboost import CatBoostRegressor, Pool
import numpy as np
import shap
shap.initjs()
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns


class Catmodel:
    """
    A class to build, train, and evaluate a CatBoost regression model with additional tools for analyzing 
    feature importance and error metrics.

    Attributes:
        name (str): Name of the model (used as the directory name for model logs).
        params (dict): Dictionary of hyperparameters for the CatBoost model.
        train_pool (Pool): Pool object for training data.
        val_pool (Pool): Pool object for validation data.
        model (CatBoostRegressor): Trained CatBoostRegressor model.
        x_val (pd.DataFrame): Validation feature set.
        y_val (pd.Series): Validation target set.
        results (pd.DataFrame): DataFrame containing validation results including predictions and errors.
        median_absolute_error (float): Median absolute error computed from validation results.
    """
    
    def __init__(self, name: str, params: dict):
        """
        Initializes the Catmodel object with a name and hyperparameters for the CatBoost model.

        Args:
            name (str): Name of the model (used for logging and identification).
            params (dict): Dictionary containing the CatBoost hyperparameters.

        Example:
            >>> params = {'iterations': 1000, 'learning_rate': 0.1, 'loss': 'RMSE'}
            >>> model = Catmodel(name='sales_forecast_model', params=params)
        """
        self.name = name
        self.params = params
    
    def set_data_pool(self, train_pool: Pool, val_pool: Pool):
        """
        Sets the training and validation data pools for the model.

        Args:
            train_pool (Pool): Pool object containing training data.
            val_pool (Pool): Pool object containing validation data.

        Example:
            >>> model.set_data_pool(train_pool, val_pool)
        """
        self.train_pool = train_pool
        self.val_pool = val_pool
    
    def set_data(self, X: pd.DataFrame, y: pd.Series, day_horizon: str, weight: np.ndarray = None):
        """
        Prepares training and validation data pools by splitting the dataset based on a date horizon.

        Args:
            X (pd.DataFrame): Feature dataset.
            y (pd.Series): Target variable dataset.
            day_horizon (str): Date threshold to split the training and validation data.
            weight (np.ndarray, optional): Optional sample weights for each row.

        Example:
            >>> model.set_data(X, y, day_horizon='2023-09-01')
        """
        cat_features = X.select_dtypes(include=['object']).columns.tolist()
        for col in cat_features:
            X[col] = X[col].astype(str)
        cat_features_idx = X.select_dtypes(include=['object']).columns.tolist()
        x_train, self.x_val = X.loc[X.date < day_horizon], X.loc[X.date >= day_horizon]
        y_train, self.y_val = y.loc[X.date < day_horizon], y.loc[X.date >= day_horizon]
        self.train_pool = Pool(x_train, y_train, cat_features=cat_features_idx, weight=weight)
        self.val_pool = Pool(self.x_val, self.y_val, cat_features=cat_features_idx, weight=weight)
    
    def prepare_model(self):
        """
        Prepares the CatBoostRegressor model with the specified hyperparameters from `self.params`.

        Example:
            >>> model.prepare_model()
        """
        self.model = CatBoostRegressor(
                loss_function=self.params['loss'],
                random_seed=self.params['seed'],
                logging_level='Silent',
                iterations=self.params['iterations'],
                max_depth=self.params['max_depth'],
                eval_metric=self.params['eval_metric'],
                learning_rate=self.params['learning_rate'],
                od_type='Iter',
                od_wait=40,
                train_dir=self.name,
                has_time=True
        )
    
    def learn(self, plot: bool = False):
        """
        Trains the CatBoost model using the training pool, with optional plotting of the learning curve.

        Args:
            plot (bool, optional): If True, plots the learning curve during training. Defaults to False.

        Example:
            >>> model.learn(plot=True)
        """
        self.prepare_model()
        self.model.fit(self.train_pool, eval_set=self.val_pool, plot=plot)
        print(f"{self.name}, early-stopped model tree count: {self.model.tree_count_}")
    
    def score(self) -> float:
        """
        Scores the model on the validation pool using the built-in CatBoost evaluation metric.

        Returns:
            float: The score of the model on the validation data.

        Example:
            >>> val_score = model.score()
            >>> print(f"Validation Score: {val_score}")
        """
        return self.model.score(self.val_pool)
    
    def show_importances(self, kind: str = "bar"):
        """
        Plots the SHAP feature importance for the validation data.

        Args:
            kind (str, optional): Type of SHAP plot. Can be "bar" or "dot". Defaults to "bar".

        Example:
            >>> model.show_importances(kind="bar")
        """
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.val_pool)
        if kind == "bar":
            return shap.summary_plot(shap_values, self.x_val, plot_type="bar")
        return shap.summary_plot(shap_values, self.x_val)
    
    def get_val_results(self):
        """
        Computes predictions and errors for the validation dataset and stores them in a DataFrame.

        Example:
            >>> model.get_val_results()
        """
        self.results = pd.DataFrame(self.y_val)
        self.results["prediction"] = self.predict(self.x_val)
        self.results["error"] = np.abs(self.results[self.results.columns.values[0]].values - self.results.prediction)
        self.results["month"] = self.x_val.month
        self.results["SquaredError"] = self.results.error.apply(lambda l: np.power(l, 2))
    
    def show_val_results(self):
        """
        Visualizes validation results with error distribution and scatter plot of predictions vs target values.

        Returns:
            plt.Axes: The axes containing the error distribution and scatter plots.

        Example:
            >>> model.show_val_results()
        """
        self.get_val_results()
        fig, ax = plt.subplots(1, 2, figsize=(20, 5))
        sns.displot(self.results.error, ax=ax[0])
        ax[0].set_xlabel("Single absolute error")
        ax[0].set_ylabel("Density")
        self.median_absolute_error = np.median(self.results.error)
        print(f"Median absolute error: {self.median_absolute_error}")
        ax[0].axvline(self.median_absolute_error, c="black")
        ax[1].scatter(self.results.prediction.values,
                      self.results[self.results.columns[0]].values,
                      c=self.results.error, cmap="RdYlBu_r", s=1)
        ax[1].set_xlabel("Prediction")
        ax[1].set_ylabel("Target")
        return ax
    
    def get_monthly_RMSE(self) -> pd.Series:
        """
        Computes the Root Mean Squared Error (RMSE) for each month in the validation dataset.

        Returns:
            pd.Series: The monthly RMSE values.

        Example:
            >>> monthly_rmse = model.get_monthly_RMSE()
            >>> print(monthly_rmse)
        """
        return self.results.groupby("month").SquaredError.mean().apply(lambda l: np.sqrt(l))
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts target values for the given feature set using the trained CatBoost model.

        Args:
            X (pd.DataFrame): Feature set for which predictions are to be made.

        Returns:
            np.ndarray: The predicted target values.

        Example:
            >>> predictions = model.predict(X_val)
        """
        return self.model.predict(X)
    
    def get_dependence_plot(self, feature1: str, feature2: str = None):
        """
        Plots SHAP dependence plot for the given feature, optionally showing interactions with another feature.

        Args:
            feature1 (str): The main feature for which the dependence plot is generated.
            feature2 (str, optional): The feature to show interactions with. Defaults to None.

        Example:
            >>> model.get_dependence_plot('price_mode')
        """
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.val_pool)
        if feature2 is None:
            return shap.dependence_plot(feature1, shap_values, self.x_val)
        else:
            return shap.dependence_plot(feature1, shap_values, self.x_val, interaction_index=feature2)