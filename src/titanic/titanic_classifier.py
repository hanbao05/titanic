import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier

class TitanicClassifier:
    def __init__(
            self,
            n_estimators=10,
            max_depth=3,
            random_state=42
        ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def output_csv(self, passenger_ids, predictions, filename):
        predictions_df = pd.DataFrame({ 'PassengerId': passenger_ids, 'Survived': predictions })
        predictions_df.to_csv(filename, index=False)