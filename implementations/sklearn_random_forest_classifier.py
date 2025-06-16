import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
import joblib 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import time
import json
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from player_stats import Player


# =========================


def run_grid_search(X, y):
    param_grid = {
        'n_estimators': [100, 150, 200, 250, 300],
        'max_depth': [10, 15, 20, 25, 30],
        'min_samples_split': [2, 3, 5, 7, 10]
    }
    clf = RandomForestClassifier()
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    print("best parameters:", grid_search.best_params_)
    print("best cross val acc:", grid_search.best_score_)
    return grid_search

def run_randomized_search(X, y, n_iter=100, cv=5):
    param_distributions = {
        'n_estimators': randint(100, 300),
        'max_depth': randint(10, 30),
        'min_samples_split': randint(2, 10)
    }
    
    clf = RandomForestClassifier()
    
    search = RandomizedSearchCV(
        clf,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )
    
    search.fit(X, y)
    
    print("Best parameters:", search.best_params_)
    print("Best CV accuracy:", search.best_score_)
    
    return search

def run_normal_model_cross_val_random_search_and_save(X, y, n_iter=100, cv=5, test_size=0.10, random_state=42):
    '''
        full cross-val, random search, eval and save model
        X: features, which are encoded if needed for categorical features
        y: labels
        n_iter: number of random search iterations
        cv: cross-validation folds
        test_size: test split size
        random_state: seed
    '''

    model_name = "elo_with_complimentary_model"
    columns_name = "elo_with_complimentary_columns"

    # save columns when testing with new data in the future
    joblib.dump(X.columns.tolist(), f"models/{columns_name}.pkl")

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    param_distributions = {
        'n_estimators': randint(100, 250),
        'max_depth': randint(5, 20),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5),
        'max_features': ['sqrt', 'log2', 0.5],
        'bootstrap': [True, False]
    }

    clf = RandomForestClassifier(random_state=random_state)

    search = RandomizedSearchCV(
        clf,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2,
        random_state=random_state
    )

    start = time.time()
    search.fit(X_train, y_train)
    end = time.time()

    print(f"\nTotal search time: {(end - start)/60:.2f} minutes")
    print("\nBest parameters:", search.best_params_)
    print("\nBest cross-val acc:", search.best_score_)

    # eval
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print("\nTest set accuracy:", test_acc)

    # save
    joblib.dump(best_model, f"models/{model_name}.pkl")

def run_minimal_model_cross_val_random_search_and_save(X, y, n_iter=100, cv=5, test_size=0.10):
    
    model_name = "elo_only_model"
    columns_name = "elo_only_columns"

    # Save feature columns
    joblib.dump(X.columns.tolist(), f"models/{columns_name}.pkl")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    # Random search parameter space
    param_distributions = {
        'n_estimators': randint(100, 250),
        'max_depth': randint(5, 20),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5),
        'max_features': ['sqrt', 'log2', 0.5],
        'bootstrap': [True, False]
    }

    clf = RandomForestClassifier()

    search = RandomizedSearchCV(
        clf,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2,
    )

    start = time.time()
    search.fit(X_train, y_train)
    end = time.time()

    print(f"\nTotal search time: {(end - start)/60:.2f} minutes")
    print("\nBest parameters:", search.best_params_)
    print("\nBest cross-val accuracy:", search.best_score_)

    # eval
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print("Test accuracy:", test_acc)

    # save
    joblib.dump(best_model, f"models/{model_name}.pkl")

def run_simple_cross_val(X, y, n_splits=5):
    clf = RandomForestClassifier()
    scores = cross_val_score(clf, X, y, cv=n_splits, scoring='accuracy')
    
    print(f"\nCross-validation scores: {scores}")
    print(f"Mean CV accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
    
    return scores

# =========================

def encode_categorical_features(X):
    categorical_cols = ['player1_hand', 'player2_hand', 'surface', 'tourney_level', 'best_of']
    
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype('category')
    
    X = pd.get_dummies(X, columns=categorical_cols)

    round_order = {
        'R128': 1,
        'R64': 2,
        'R32': 3,
        'R16': 4,
        'QF': 5,
        'SF': 6,
        'F': 7
    }

    X['tourney_round'] = X['tourney_round'].map(round_order)

    return X


# =========================

# dataset
df = pd.read_csv("datasets/elo_driven_features_2010_2024.csv")

# split features and labels
X = df.drop("player1_won", axis=1)
y = df["player1_won"]

# encode
X = encode_categorical_features(X)

minimal_features = [
    'player1_welo', 'player2_welo',
    'player1_surface_welo', 'player2_surface_welo',
    'player1_won'
]
df_minimal = df[minimal_features]
X_minimal = df_minimal.drop("player1_won", axis=1)
y_minimal = df_minimal["player1_won"]


#run_minimal_model_cross_val_random_search_and_save(X_minimal, y_minimal)





