# Titanic Survival Classifier

A project in order to analyze and fit a classifier model to the famous Titanic Survival Dataset

Kaggle: https://www.kaggle.com/competitions/titanic/overview

## Installation

Make sure that you have the correct Python version is installed `>= 3.8`.

In order to install the required Python packages, it's recommended to setup a Python virtual environment at first

```
python -m venv .venv
```

Activate the virtual environment

```
source .venv/bin/activate
```

Install the required dependencies from `requirements.txt`, (Make sure the virtual environment is installed)

```
pip install -r requirements.txt
```

## Usage

```python
from src.titanic import TitanicClassifier, TitanicDataset

raw_train_df = pd.read_csv('data/train.csv')
raw_test_df = pd.read_csv('data/test.csv')

dataset = TitanicDataset()
train_df, test_df = dataset.preprocess(raw_train_df, raw_test_df)
dataset.heatmap(train_df)
dataset.visualize(train_df)
```

```
X_train = train_df.drop(columns=['Survived'])
y_train = train_df['Survived']

X_test = test_df

classifier = TitanicClassifier()
classifier.train(X_train, y_train)

predictions = classifier.predict(X_test)
```

## License

This project is [MIT Licensed](./LICENSE.txt)