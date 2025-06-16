import pandas as pd
import random
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
        

def build_tree(dataset, max_depth, min_size):
    root = get_split(dataset)
    split(root, max_depth, min_size, 1)
    return root

def terminal_node(group) -> Node:
    class_values = list(set(row[-1] for row in group))
    
    gini = gini_index(group, [], class_values)
    s = len(group)
    d = [sum(1 for row in group if row[-1] == value) for value in class_values]
    p = class_values[d.index(max(d))]
    
    return Node(
        gini_self=gini,
        samples=s,
        distribution=d,
        predicted_class=p
    )
    
def split(node, max_depth, min_size, depth):
    left_rows, right_rows = node.left, node.right

    # Stop if one side is empty
    if not left_rows or not right_rows:
        node.left = terminal_node(left_rows + right_rows)
        node.right = terminal_node(left_rows + right_rows)
        return

    if all(row[-1] == left_rows[0][-1] for row in left_rows):
        node.left = terminal_node(left_rows)
    elif len(left_rows) <= min_size or depth >= max_depth:
        node.left = terminal_node(left_rows)
    else:
        node.left = get_split(left_rows)
        split(node.left, max_depth, min_size, depth + 1)

    if all(row[-1] == right_rows[0][-1] for row in right_rows):
        node.right = terminal_node(right_rows)
    elif len(right_rows) <= min_size or depth >= max_depth:
        node.right = terminal_node(right_rows)
    else:
        node.right = get_split(right_rows)
        split(node.right, max_depth, min_size, depth + 1)

def gini_index(left_group, right_group, class_values):
    total_samples = len(left_group) + len(right_group)
    gini = 0.0


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
     
def get_split(dataset) -> Node:
    class_values = list(set(row[-1] for row in dataset))

    best_index, best_value = None, None
    best_gini = float('inf')
    best_left, best_right = None, None
    gini_self = gini_index(dataset, [], class_values)
    

    row_length = len(dataset[0])
    for index in range(0, row_length - 1):  #loop on features
        
        midpoints = sorted(set(row[index] for row in dataset))
        
        for i in range(0, len(midpoints) - 1):
            
            midpoint = (midpoints[i] + midpoints[i + 1]) / 2.
            left, right = split_dataset(index, midpoint, dataset)
            gini = gini_index(left, right, class_values)

            if gini < best_gini:
                best_index, best_value = index, midpoint 
                best_gini = gini
                best_left, best_right = left, right

    s = len(dataset)
    d = [sum(1 for row in dataset if row[-1] == value) for value in class_values]
    p = class_values[d.index(max(d))]
    
    return Node(
        index = best_index,
        value = best_value,
        
        gini_split = best_gini,
        gini_self = gini_self,
        
        left = best_left,
        right = best_right,

        samples = s,
        distribution = d,
        predicted_class = p      
    )

def split_dataset(index, value, dataset):
    left, right = [], []
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

def predict(node, row):
    while node.index is not None:
        if row[node.index] < node.value:
            node = node.left
        else:
            node = node.right
    return node.predicted_class

def print_tree(node, depth=0, feature_names=None):
    indent = "     " * depth
    if node.index is not None:
        feature = f"X{node.index}" if not feature_names else feature_names[node.index]
        print(f"{indent}[{feature} < {node.value:.2f}] (gini_split={node.gini_split:.3f}, gini_self= {node.gini_self:.3f}, distribution = {node.distribution}, samples={node.samples}, class={node.predicted_class})")
        print_tree(node.left, depth + 1, feature_names)
        print_tree(node.right, depth + 1, feature_names)
    else:
        print(f"{indent}[Leaf] (gini_self={node.gini_self:.3f}, distribution= {node.distribution},samples={node.samples}, class={node.predicted_class})")

def test_tree_acc(tree, data):
    correct = 0
    for row in data:
        if predict(tree, row) == row[-1]:
            correct += 1
    return correct / len(data) * 100



# ====================

def read_Sonar():
    feature_names = [f'Feature{i+1}' for i in range(60)] + ['Label']
    df = pd.read_csv('datasets/practice_datasets/sonar.all-data.csv', header=None, names=feature_names)
    
    dataset = df.values.tolist()
    random.shuffle(dataset)
    
    return dataset



# =======================

dataset = read_Sonar()

train_split = int(0.7 * len(dataset))

train_data = dataset[:train_split]
val_data = dataset[train_split:]

# =======================

#config
max_depth = 10
min_size = 1

# ================

start = time.time()
tree = build_tree(train_data, max_depth=max_depth, min_size=min_size)

print(f"Decesion Tree built in {time.time() - start:.2f} seconds")

# =======================

val_accuracy = test_tree_acc(tree, val_data)
print(f"Decision Tree Val Acc: {val_accuracy:.4f}%")






