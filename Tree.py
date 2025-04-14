import pandas as pd
import numpy as np
from graphviz import Digraph

def load_data(file_path):
    data = pd.read_csv(file_path, header=None)
    conditional_attribute = data.iloc[:, :-1]  #kolumny oprócz ostatniej
    last_column = data.iloc[:, -1]  #ostatnia kolumna
    return conditional_attribute, last_column

def calculate_entropy(last_column):
    value_counts = last_column.value_counts()  # Policz wystąpienia każdej wartości
    probabilities = value_counts / len(last_column)  # Oblicz prawdopodobieństwa
    entropy = -np.sum(probabilities * np.log2(probabilities))  # Oblicz entropię
    return entropy

def calculate_information_gain(conditional_attribute, last_column, column_index):
    total_entropy = calculate_entropy(last_column)  # Entropia całego zbioru
    values, counts = np.unique(conditional_attribute.iloc[:, column_index], return_counts=True)  # Unikalne wartości i ich liczności
    # Oblicz entropię warunkową
    weighted_entropy = np.sum([(counts[i] / np.sum(counts)) * calculate_entropy(last_column[conditional_attribute.iloc[:, column_index] == values[i]]) for i in range(len(values))])
    information_gain = total_entropy - weighted_entropy  # Zysk informacyjny to różnica entropii
    return information_gain

def calculate_information_value(conditional_attribute, last_column, column_index):
    values, counts = np.unique(conditional_attribute.iloc[:, column_index], return_counts=True)  # Unikalne wartości i ich liczności
    # Oblicz entropię warunkową
    weighted_entropy = np.sum([(counts[i] / np.sum(counts)) * calculate_entropy(last_column[conditional_attribute.iloc[:, column_index] == values[i]]) for i in range(len(values))])
    return weighted_entropy

def calculate_split_info(conditional_attribute, column_index):
    values, counts = np.unique(conditional_attribute.iloc[:, column_index], return_counts=True)  # Unikalne wartości i ich liczności
    probabilities = counts / np.sum(counts)  # Oblicz prawdopodobieństwa
    split_info = -np.sum(probabilities * np.log2(probabilities))  # Oblicz SplitInfo
    return split_info

def calculate_gain_ratio(conditional_attribute, last_column, column_index):
    info_gain = calculate_information_gain(conditional_attribute, last_column, column_index)  # Oblicz zysk informacyjny
    split_info = calculate_split_info(conditional_attribute, column_index)  # Oblicz SplitInfo
    if split_info == 0:  # Jeśli SplitInfo wynosi 0, zwróć 0
        return 0
    gain_ratio = info_gain / split_info  # Gain Ratio to stosunek zysku informacyjnego do SplitInfo
    return gain_ratio

def select_best_attribute(conditional_attribute, last_column):
    best_gain_ratio = -1  # Inicjalizacja najlepszego Gain Ratio
    best_feature_index = -1  # Inicjalizacja indeksu najlepszego atrybutu
    for feature_index in range(conditional_attribute.shape[1]):  # Iteracja po wszystkich atrybutach
        gain_ratio = calculate_gain_ratio(conditional_attribute, last_column, feature_index)  # Oblicz Gain Ratio
        if gain_ratio > best_gain_ratio:  # Jeśli Gain Ratio jest lepsze, zaktualizuj
            best_gain_ratio = gain_ratio
            best_feature_index = feature_index
    return best_feature_index

class TreeNode:
    def __init__(self, column_index=None, value=None, children=None, *, label=None):
        # Inicjalizacja węzła drzewa
        self.feature_index = column_index  # Indeks atrybutu
        self.value = value  # Wartość atrybutu
        self.children = children if children is not None else {}  # Dzieci węzła
        self.label = label  # Etykieta węzła (dla liści)

def build_decision_tree(conditional_attribute, last_column, column_index=None):
    node = TreeNode()  # Utwórz nowy węzeł
    if len(np.unique(last_column)) > 1 and (column_index is None or len(column_index) > 0):  # Sprawdź warunek stopu
        if column_index is None:  # Jeśli brak ograniczeń, użyj wszystkich cech
            column_index = list(range(conditional_attribute.shape[1]))
        best_feature_index = select_best_attribute(conditional_attribute.iloc[:, column_index], last_column)  # Wybierz najlepszy atrybut
        best_feature = column_index[best_feature_index]  # Pobierz indeks najlepszego atrybutu
        node.feature_index = best_feature  # Ustaw atrybut w węźle
        feature_values = np.unique(conditional_attribute.iloc[:, best_feature])  # Pobierz unikalne wartości atrybutu
        for value in feature_values:  # Iteracja po wartościach atrybutu
            sub_X = conditional_attribute[conditional_attribute.iloc[:, best_feature] == value]  # Podzbiór danych dla danej wartości
            sub_y = last_column[conditional_attribute.iloc[:, best_feature] == value]  # Podzbiór etykiet
            new_feature_indices = [i for i in column_index if i != best_feature]  # Usuń użyty atrybut
            child_node = build_decision_tree(sub_X, sub_y, new_feature_indices)  # Rekurencyjnie buduj drzewo
            node.children[value] = child_node  # Dodaj dziecko do węzła
    else:
        node.label = last_column.mode()[0]  # Jeśli warunek stopu, ustaw etykietę węzła
    return node  # Zwróć węzeł

def save_tree_to_dot(node, filename):
    def add_nodes_edges(dot, node, parent=None, edge_label=""):
        if node is None:
            return
        if node.label is not None:  # Jeśli węzeł jest liściem
            dot.node(str(id(node)), f"Leaf: {node.label}", shape="box")  # Dodaj węzeł liścia
        else:
            dot.node(str(id(node)), f"Kolumna {node.feature_index+1}")  # Dodaj węzeł atrybutu
        if parent is not None:  # Jeśli istnieje rodzic, dodaj krawędź
            dot.edge(str(id(parent)), str(id(node)), label=edge_label)
        for value, child in node.children.items():  # Iteracja po dzieciach
            add_nodes_edges(dot, child, node, str(value))  # Rekurencyjnie dodaj dzieci

    dot = Digraph()  # Utwórz obiekt Digraph
    add_nodes_edges(dot, node)  # Dodaj węzły i krawędzie
    dot.save(filename)

if __name__ == "__main__":
    file_path = 'car.data'
    conditional_attribute, last_column = load_data(file_path)
    entropy = calculate_entropy(last_column)
    print(f"Info(T): {entropy}")
    print()
    for column_index in range(conditional_attribute.shape[1]):
        info_value = calculate_information_value(conditional_attribute, last_column, column_index)
        print(f"Info(a{column_index}, T): {info_value}")
    print()
    for column_index in range(conditional_attribute.shape[1]):
        info_gain = calculate_information_gain(conditional_attribute, last_column, column_index)
        print(f"Gain(a{column_index}, T): {info_gain}")
    print()
    for column_index in range(conditional_attribute.shape[1]):
        split_info = calculate_split_info(conditional_attribute, column_index)
        print(f"SplitInfo(a{column_index}, T): {split_info}")
    print()
    for column_index in range(conditional_attribute.shape[1]):
        gain_ratio = calculate_gain_ratio(conditional_attribute, last_column, column_index)
        print(f"GainRatio(a{column_index}, T): {gain_ratio}")
    print()
    best_attribute = select_best_attribute(conditional_attribute, last_column)
    print(f"Najlepszy atrybut do podziału: a{best_attribute}")
    decision_tree = build_decision_tree(conditional_attribute, last_column)
    save_tree_to_dot(decision_tree, "decision_tree.dot")
