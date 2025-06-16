import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


df = pd.read_csv("datasets/elo_driven_features_2010_2024.csv")

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

X = df.drop("player1_won", axis=1)
y = df["player1_won"]

X = encode_categorical_features(X)


# Split
test_size = 0.10
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

# =========================

# Train Decision Tree
clf = DecisionTreeClassifier(max_depth=15, min_samples_split=5)
clf.fit(X_train, y_train)

#y_pred = clf.predict(X_test)
#print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))
