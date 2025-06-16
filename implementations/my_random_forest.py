import random
import pandas as pd
from tqdm import tqdm
import time
from sklearn.preprocessing import LabelEncoder

#random.seed(40)

class Node():
    def __init__(self, 
        index = None, value = None, 
        gini_split = None, gini_self = None,
        left = None, right = None, 
        samples = None, distribution = None, predicted_class = None,  
        ):
        self.index = index
        self.value = value
        
        self.gini_split = gini_split
        self.gini_self = gini_self

        self.left = left
        self.right = right

        self.samples = samples
        self.distribution = distribution
        self.predicted_class = predicted_class



def subset(dataset, ratio):
    n_subset = round(len(dataset) * ratio)
    return random.choices(dataset, k=n_subset)

#same rows can happen yes
   
def calc_gini_for_one_group(group, class_values):
    total_samples = len(group)
    gini = 1.0

    if total_samples == 0:
        return 0.0  

    score = 0.0
    for class_value in class_values:
        proportion = [row[-1] for row in group].count(class_value) / total_samples
        score += proportion ** 2

    return gini - score

def calc_gini_index_for_a_split(left_group, right_group, class_values):
    total_samples = len(left_group) + len(right_group)
    gini = 0.0
    score = 0.0

    #left group
    size_left = len(left_group)
    if size_left > 0:
        score = 0.0
        for class_value in class_values:
            proportion = [row[-1] for row in left_group].count(class_value) / size_left
            score += proportion ** 2
        gini += (1.0 - score) * (size_left / total_samples)

    #right group
    size_right = len(right_group)
    if size_right > 0:
        score = 0.0
        for class_value in class_values:
            proportion = [row[-1] for row in right_group].count(class_value) / size_right
            score += proportion ** 2
        gini += (1.0 - score) * (size_right / total_samples)

    return gini 
    
def split_on_val_and_ind(index, value, dataset):
    left, right = [], []
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right    
    
    
def get_best_possible_split_for_n_features(dataset, n_features=None) -> Node:
    class_values = list(set(row[-1] for row in dataset)) #unique labels
    
    best_index, best_value = None, None
    best_gini = float('inf')
    best_left, best_right = None, None
    gini_self = calc_gini_for_one_group(dataset, class_values)

    #if no n_feature is given just use everything 
    feature_indices = list(range(len(dataset[0]) - 1))
    if n_features:
        feature_indices = random.sample(feature_indices, n_features)

    for index in feature_indices:
        #more optimal with midpoints and also sklearn uses that also so dont actually have to split on concrete values
        midpoints = sorted(set(row[index] for row in dataset))
        
        #looking for the best possible split
        for i in range(len(midpoints) - 1):
            midpoint = (midpoints[i] + midpoints[i + 1]) / 2.0
            left, right = split_on_val_and_ind(index, midpoint, dataset)
            gini = calc_gini_index_for_a_split(left, right, class_values)

            if gini < best_gini:
                best_index, best_value = index, midpoint 
                best_gini = gini
                best_left, best_right = left, right

    s = len(dataset)
    d = [sum(1 for row in dataset if row[-1] == value) for value in class_values]
    p = class_values[d.index(max(d))]

    return Node(
        index=best_index,
        value=best_value,
        gini_split=best_gini,
        gini_self=gini_self,
        left=best_left,
        right=best_right,
        samples=s,
        distribution=d,
        predicted_class=p
    )
   
   
#change: passing left and right rows also to avoid NoneType + NoneType
def split(node, max_depth, min_size, depth, n_features, left_rows, right_rows):
    
    # left_rows, right_rows = node.left, node.right

    #mine
    #a split happened where all of the rows went to either left or right
    #if not left_rows or not right_rows:
    #    node.left = terminal_node(left_rows + right_rows)
    #    node.right = terminal_node(left_rows + right_rows)
    #    return

    #solution for noneType w list fallback
    if not left_rows or not right_rows:
        left_rows = left_rows if left_rows else []
        right_rows = right_rows if right_rows else []
        
        combined = left_rows + right_rows
        node.left = terminal_node(combined)
        node.right = terminal_node(combined)
        return

    # 1) all labels/class values match
    # 2) reached below min size for a terminal node or tree reached the depth given at the start and wont split anymore
    # 3) keep splitting aa
    if all(row[-1] == left_rows[0][-1] for row in left_rows):
        node.left = terminal_node(left_rows)
    elif len(left_rows) <= min_size or depth >= max_depth:
        node.left = terminal_node(left_rows)
    else:
        node.left = get_best_possible_split_for_n_features(left_rows, n_features)
        split(node.left, max_depth, min_size, depth + 1, n_features, node.left.left, node.left.right)

    #same
    if all(row[-1] == right_rows[0][-1] for row in right_rows):
        node.right = terminal_node(right_rows)
    elif len(right_rows) <= min_size or depth >= max_depth:
        node.right = terminal_node(right_rows)
    else:
        node.right = get_best_possible_split_for_n_features(right_rows, n_features)
        split(node.right, max_depth, min_size, depth + 1, n_features, node.right.left, node.right.right)
   
   
def terminal_node(group) -> Node:
    if not group:
        return Node(
            gini_self=0.0,
            samples=0,
            distribution=[],
            predicted_class=None 
    )

    class_values = list(set(row[-1] for row in group))
    
    gini = calc_gini_for_one_group(group, class_values)
    s = len(group)
    d = [sum(1 for row in group if row[-1] == value) for value in class_values] 
    p = class_values[d.index(max(d))]
    
    return Node(
        gini_self=gini,
        samples=s,
        distribution=d,
        predicted_class=p
    )
   
   
#with new split
def build_tree_with_n_features(dataset, max_depth, min_size, n_features):
    root = get_best_possible_split_for_n_features(dataset, n_features)
    left_rows, right_rows = root.left, root.right
    split(root, max_depth, min_size, 1, n_features, left_rows, right_rows)

    return root



def build_forest(dataset, n_trees, ratio_of_trees, max_depth, min_size, n_features):
    forest = []
    print("starting to build forest..")

    for _ in tqdm(range(n_trees), desc="Building Forest"):
        sample = subset(dataset, ratio_of_trees)
        tree = build_tree_with_n_features(sample, max_depth, min_size, n_features)
        forest.append(tree)
    return forest


def tree_predict_class(node, row):
    while node.index is not None:
        if row[node.index] < node.value:
            node = node.left
        else:
            node = node.right
    return node.predicted_class


def forest_predict(forest, row):
    voted_classes = [tree_predict_class(tree, row) for tree in forest]

    if not voted_classes:
        return 0 
    
    voted_classes_dict = {}
    for v in voted_classes:
        voted_classes_dict[v] = voted_classes_dict.get(v, 0) + 1

    most_common = max(voted_classes_dict, key=voted_classes_dict.get)

    return most_common


def test_forest_acc(forest, data):
    correct = 0
    for row in tqdm(data, desc="Evaluating Forest"):
        if forest_predict(forest, row) == row[-1]:
            correct += 1
    return correct / len(data) * 100

#===================================================



def read_Sonar():
    feature_names = [f'Feature{i+1}' for i in range(60)] + ['Label']
    df = pd.read_csv('datasets/practice_datasets/sonar.all-data.csv', header=None, names=feature_names)
    
    dataset = df.values.tolist()
    random.shuffle(dataset)
    
    return dataset


    

# ======================

dataset = read_Sonar()


train_split = int(0.5 * len(dataset))
val_split = int(0.75 * len(dataset))

train_data = dataset[:train_split]
val_data = dataset[train_split:val_split]
test_data = dataset[val_split:]



# ==============================


#configs
tree_count = 100
ratio_of_trees = 0.7
max_depth = 10
min_size = 1
n_features = int((len(dataset[0]) - 1) * 0.5)




# ====================


start = time.time()
forest = build_forest(train_data, tree_count, ratio_of_trees, max_depth, min_size, n_features)

print(f"forest built in {time.time() - start:.2f} seconds")


# ====================


val_accuracy = test_forest_acc(forest, val_data)
print(f"Random forest Val Acc: {val_accuracy:.4f}%")

test_accuracy = test_forest_acc(forest, test_data)
print(f"Random Forest Test Acc: {test_accuracy:.4f}%")



