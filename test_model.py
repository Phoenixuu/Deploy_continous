import unittest
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd


class TestRegressionModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load data
        iowa_file_path = 'train.csv'
        home_data = pd.read_csv(iowa_file_path)
        
        # Define target (y) and features (X)
        y = home_data.SalePrice
        features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
        X = home_data[features]

        # Split data
        cls.train_X, cls.val_X, cls.train_y, cls.val_y = train_test_split(X, y, random_state=1)

    def evaluate_model(self, model):
        # Fit the model
        model.fit(self.train_X, self.train_y)
        # Predict and calculate MAE
        val_predictions = model.predict(self.val_X)
        val_mae = mean_absolute_error(val_predictions, self.val_y)
        return val_mae

    def test_decision_tree_regressor_no_max_leaf_nodes(self):
        model = DecisionTreeRegressor(random_state=1)
        val_mae = self.evaluate_model(model)
        self.assertLess(val_mae, 30000, f"MAE should be less than 30000. Current value: {val_mae}")

    def test_decision_tree_regressor_with_max_leaf_nodes(self):
        model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
        val_mae = self.evaluate_model(model)
        self.assertLess(val_mae, 28000, f"MAE should be less than 28000. Current value: {val_mae}")

    def test_random_forest_regressor(self):
        model = RandomForestRegressor(random_state=1)
        val_mae = self.evaluate_model(model)
        self.assertLess(val_mae, 23000, f"MAE should be less than 23000. Current value: {val_mae}")


if __name__ == '__main__':
    unittest.main()

