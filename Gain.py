import pandas as pd
import numpy as np
from InfoXT import oblicz_info


def oblicz_przyrost_informacji(df, kolumna):
    """
    Oblicza przyrost informacji dla podanego atrybutu.
    :param df: DataFrame zawierający dane
    :param kolumna: Nazwa kolumny, dla której obliczamy przyrost informacji
    :return: Wartość przyrostu informacji
    """
    entropia_pelna = -sum((count / len(df)) * np.log2(count / len(df))
                          for count in df.iloc[:, -1].value_counts().to_dict().values())
    info_atrybut = oblicz_info(df, kolumna)
    return entropia_pelna - info_atrybut


if __name__ == "__main__":
    # Wczytanie danych
    plik_kategorii = "gielda.txt"
    df_kategorie = pd.read_csv(plik_kategorii, sep=",", header=None)

    # Obliczenie przyrostu informacji dla każdej kolumny (bez ostatniej decyzyjnej)
    for col in df_kategorie.columns[:-1]:
        gain = oblicz_przyrost_informacji(df_kategorie, col)
        print(f"Przyrost informacji dla atrybutu {col}: {gain}")