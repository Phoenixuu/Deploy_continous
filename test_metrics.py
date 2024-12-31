import unittest
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd


class TestMetrics(unittest.TestCase):
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

    def test_model_metrics(self):
        model = RandomForestRegressor(random_state=1)
        model.fit(self.train_X, self.train_y)
        predictions = model.predict(self.val_X)

        # Calculate metrics
        mae = mean_absolute_error(self.val_y, predictions)
        mse = mean_squared_error(self.val_y, predictions)
        r2 = r2_score(self.val_y, predictions)

        # Assertions
        self.assertLess(mae, 23000, f"MAE should be less than 23000. Current value: {mae}")
        self.assertLess(mse, 1e9, f"MSE should be less than 1e9. Current value: {mse}")
        self.assertGreater(r2, 0.7, f"R2 Score should be greater than 0.7. Current value: {r2}")


if __name__ == '__main__':
    unittest.main()

