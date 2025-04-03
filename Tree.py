import pandas as pd
import numpy as np
from collections import deque
from graphviz import Digraph
import shutil  # Dodaj import

def load_data(file_path):
    """
    Wczytuje dane z pliku i zwraca DataFrame.
    Ostatnia kolumna jest traktowana jako atrybut decyzyjny.
    """
    data = pd.read_csv(file_path, header=None)
    X = data.iloc[:, :-1]  # Wszystkie kolumny oprócz ostatniej
    y = data.iloc[:, -1]   # Ostatnia kolumna
    return X, y

def calculate_entropy(y):
    """
    Oblicza entropię dla atrybutu decyzyjnego.
    """
    value_counts = y.value_counts()
    probabilities = value_counts / len(y)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

def calculate_information_gain(X, y, feature_index):
    """
    Oblicza zysk informacyjny dla danego atrybutu (kolumny) w X.
    """
    # Oblicz entropię całego zbioru
    total_entropy = calculate_entropy(y)
    
    # Oblicz entropię warunkową dla danego atrybutu
    values, counts = np.unique(X.iloc[:, feature_index], return_counts=True)
    weighted_entropy = np.sum([(counts[i] / np.sum(counts)) * calculate_entropy(y[X.iloc[:, feature_index] == values[i]]) for i in range(len(values))])
    
    # Zysk informacyjny to różnica między entropią całego zbioru a entropią warunkową
    information_gain = total_entropy - weighted_entropy
    return information_gain

def calculate_information_value(X, y, feature_index):
    """
    Oblicza wartość informacji dla danego atrybutu (kolumny) w X.
    """
    values, counts = np.unique(X.iloc[:, feature_index], return_counts=True)
    weighted_entropy = np.sum([(counts[i] / np.sum(counts)) * calculate_entropy(y[X.iloc[:, feature_index] == values[i]]) for i in range(len(values))])
    return weighted_entropy

def calculate_split_info(X, feature_index):
    """
    Oblicza SplitInfo dla danego atrybutu (kolumny) w X.
    """
    values, counts = np.unique(X.iloc[:, feature_index], return_counts=True)
    probabilities = counts / np.sum(counts)
    split_info = -np.sum(probabilities * np.log2(probabilities))
    return split_info

def calculate_gain_ratio(X, y, feature_index):
    """
    Oblicza Gain Ratio dla danego atrybutu (kolumny) w X.
    """
    info_gain = calculate_information_gain(X, y, feature_index)
    split_info = calculate_split_info(X, feature_index)
    if split_info == 0:
        return 0
    gain_ratio = info_gain / split_info
    return gain_ratio

def select_best_attribute(X, y):
    """
    Wybiera najlepszy atrybut do podziału na podstawie Gain Ratio.
    """
    best_gain_ratio = -1
    best_feature_index = -1
    for feature_index in range(X.shape[1]):
        gain_ratio = calculate_gain_ratio(X, y, feature_index)
        if gain_ratio > best_gain_ratio:
            best_gain_ratio = gain_ratio
            best_feature_index = feature_index
    return best_feature_index

class TreeNode:
    def __init__(self, feature_index=None, value=None, children=None, *, label=None):
        self.feature_index = feature_index
        self.value = value
        self.children = children if children is not None else {}
        self.label = label

def build_decision_tree(X, y, feature_indices=None):
    """
    Buduje drzewo decyzyjne na podstawie danych X i y.
    """
    if feature_indices is None:
        feature_indices = list(range(X.shape[1]))
    
    # Jeśli wszystkie etykiety są takie same, zwróć liść z tą etykietą
    if len(np.unique(y)) == 1:
        return TreeNode(label=y.iloc[0])
    
    # Jeśli nie ma więcej atrybutów do podziału, zwróć liść z najczęstszą etykietą
    if len(feature_indices) == 0:
        return TreeNode(label=y.mode()[0])
    
    # Wybierz najlepszy atrybut do podziału
    best_feature_index = select_best_attribute(X.iloc[:, feature_indices], y)
    best_feature = feature_indices[best_feature_index]
    
    # Twórz nowy węzeł drzewa
    node = TreeNode(feature_index=best_feature)
    
    # Podziel dane na podzbiory na podstawie wartości najlepszego atrybutu
    feature_values = np.unique(X.iloc[:, best_feature])
    for value in feature_values:
        sub_X = X[X.iloc[:, best_feature] == value]
        sub_y = y[X.iloc[:, best_feature] == value]
        new_feature_indices = [i for i in feature_indices if i != best_feature]
        child_node = build_decision_tree(sub_X, sub_y, new_feature_indices)
        node.children[value] = child_node
    
    return node


def save_tree_to_dot(node, filename):
    """
    Zapisuje drzewo decyzyjne do pliku w formacie DOT.
    """
    def add_nodes_edges(dot, node, parent=None, edge_label=""):
        if node is None:
            return
        if node.label is not None:
            dot.node(str(id(node)), f"Leaf: {node.label}", shape="box")
        else:
            dot.node(str(id(node)), f"Kolumna {node.feature_index+1}")
        if parent is not None:
            dot.edge(str(id(parent)), str(id(node)), label=edge_label)
        for value, child in node.children.items():
            add_nodes_edges(dot, child, node, str(value))

    dot = Digraph()
    add_nodes_edges(dot, node)
    dot.save(filename)

if __name__ == "__main__":
    file_path = 'gielda.txt'
    X, y = load_data(file_path)
    
    entropy = calculate_entropy(y)
    print(f"Info(T): {entropy}")
    print()

    for feature_index in range(X.shape[1]):
        info_value = calculate_information_value(X, y, feature_index)
        print(f"Info(a{feature_index}, T): {info_value}")
    print()

    for feature_index in range(X.shape[1]):
        info_gain = calculate_information_gain(X, y, feature_index)
        print(f"Gain(a{feature_index}, T): {info_gain}")
    print()

    for feature_index in range(X.shape[1]):
        split_info = calculate_split_info(X, feature_index)
        print(f"SplitInfo(a{feature_index}, T): {split_info}")
    print()

    for feature_index in range(X.shape[1]):
        gain_ratio = calculate_gain_ratio(X, y, feature_index)
        print(f"GainRatio(a{feature_index}, T): {gain_ratio}")
    print()

    best_attribute = select_best_attribute(X, y)
    print(f"Najlepszy atrybut do podziału: a{best_attribute}")

    decision_tree = build_decision_tree(X, y)

    # Zapisz drzewo decyzyjne do pliku DOT
    save_tree_to_dot(decision_tree, "decision_tree.dot")

