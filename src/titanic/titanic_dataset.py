import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class TitanicDataset:
    def __init__(
            self,
            drop_threshold=0.8,
            dropped_columns=['Name', 'PassengerId', 'Ticket', 'Cabin']
        ):
        self.drop_threshold = drop_threshold
        self.dropped_columns = dropped_columns

    def cleanse(self, df):
        for col in df.columns:
            # Drop columns with too many missing values
            if df[col].isna().sum() / len(df) > self.drop_threshold:
                df = df[df[col].notna()]
            # Fill missing numerical values with the median
            elif df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].fillna(df[col].median())
            # Fill missing categorical values with the mode
            elif df[col].dtype == object:
                df[col] = df[col].fillna(df[col].mode()[0])
            # Drop rows with any remaining missing values
            else:
                df = df[df[col].notna()]

        df = df.drop(self.dropped_columns, axis=1)
        return df
    
    def map_categorical_to_numerical(self, df, columns=['Sex', 'Embarked']):
        for col in columns:
            sex_values = sorted(df[col].unique())
            sex_mapping = {val: idx for idx, val in enumerate(sex_values)}
            df[col] = df[col].map(sex_mapping)

        return df

    def preprocess(self, train_dataset, test_dataset):
        train_dataset = self.cleanse(train_dataset)
        test_dataset = self.cleanse(test_dataset)

        train_dataset = self.map_categorical_to_numerical(train_dataset)
        test_dataset = self.map_categorical_to_numerical(test_dataset)

        return train_dataset, test_dataset
    
    def heatmap(self, df):
        df = df.select_dtypes(include=[np.number])
        sns.heatmap(df.corr(), cmap='coolwarm', vmin=-1, vmax=1, annot=True)       

    def visualize(self, df):
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        ###
        #  Survived vs Sex
        ###
        survived = df['Survived'].groupby(df['Sex']).sum()
        dead = len(df) - survived
        counts = {
            "Survived": survived,
            "Dead": dead,
        }

        for i, (status, count) in enumerate(counts.items()):
            axes[0, 0].bar(df["Sex"].unique(), count, label=status, bottom=np.sum(list(counts.values())[:i], axis=0))

        axes[0, 0].set_title('Survived vs Sex')
        axes[0, 0].set_xlabel('Sex')
        axes[0, 0].set_ylabel('Survived')
        axes[0, 0].legend()

        ###
        #  Survived vs Pclass
        ###
        survived = df['Survived'].groupby(df['Pclass']).sum()
        dead = len(df) - survived
        counts = {
            "Survived": survived,
            "Dead": dead,
        }

        for i, (status, count) in enumerate(counts.items()):
            axes[0, 1].bar(df["Pclass"].unique(), count, label=status, bottom=np.sum(list(counts.values())[:i], axis=0))


        axes[0, 1].set_title('Survived vs Pclass')
        axes[0, 1].set_xlabel('Pclass')
        axes[0, 1].set_ylabel('Survived')
        axes[0, 1].legend()

        ###
        # Survived vs Fare
        ###
        axes[1, 0].hist(
            [
                df[df['Survived'] == 1]['Fare'],
                df[df['Survived'] == 0]['Fare']
            ],
            label=['Survived', 'Dead'],
        )

        axes[1, 0].set_title('Survived vs Fare')
        axes[1, 0].set_xlabel('Fare')
        axes[1, 0].set_ylabel('Survived')
        axes[1, 0].legend()

        ### 
        # Survived vs Embarked
        ###
        embarked_survived = df['Survived'].groupby(df['Embarked']).sum()
        embarked_dead = df['Embarked'].value_counts() - embarked_survived
        embarked_counts = {
            "Survived": embarked_survived,
            "Dead": embarked_dead,
        }

        for i, (status, count) in enumerate(embarked_counts.items()):
            axes[1, 1].bar(df["Embarked"].unique(), count, label=status, bottom=np.sum(list(embarked_counts.values())[:i], axis=0))

        axes[1, 1].set_title('Survived vs Embarked')
        axes[1, 1].set_xlabel('Embarked')
        axes[1, 1].set_ylabel('Survived')
        axes[1, 1].legend()

        fig.tight_layout()
        plt.show()