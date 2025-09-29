import pandas as pd

from sklearn.ensemble import RandomForestClassifier

"""
A classifier for the Titanic dataset using RandomForestClassifier from scikit-learn.
Methods:
- train: Trains the model on feature matrix X and target vector y.
- predict: Predicts survival outcomes (0 or 1) for given feature matrix X.
- output_csv: Outputs predictions to a CSV file with columns 'PassengerId' and 'Survived'.
"""
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
        """Trains the RandomForestClassifier on feature matrix X and target vector y."""
        self.model.fit(X, y)

    def predict(self, X):
        """Predicts survival outcomes (0 or 1) for the given feature matrix X."""
        return self.model.predict(X)
    
    def output_csv(self, passenger_ids, predictions, filename):
        """Outputs predictions to a CSV file with columns 'PassengerId' and 'Survived'."""
        predictions_df = pd.DataFrame({ 'PassengerId': passenger_ids, 'Survived': predictions })
        predictions_df.to_csv(filename, index=False)