import pandas as pd
import numpy as np
from Gain import oblicz_przyrost_informacji


def oblicz_wspolczynnik_split(df, kolumna):
    """
    Oblicza współczynnik split (SplitInfo) dla podanego atrybutu.
    :param df: DataFrame zawierający dane
    :param kolumna: Nazwa kolumny, dla której obliczamy współczynnik split
    :return: Wartość współczynnika split
    """
    wartosci_atrybutu = df[kolumna].value_counts(normalize=True).values
    wspolczynnik_split = -sum(p * np.log2(p) for p in wartosci_atrybutu)
    return wspolczynnik_split


def oblicz_zrownowazony_przyrost(df, kolumna):
    """
    Oblicza zrównoważony przyrost informacji dla podanego atrybutu.
    :param df: DataFrame zawierający dane
    :param kolumna: Nazwa kolumny, dla której obliczamy zrównoważony przyrost informacji
    :return: Wartość zrównoważonego przyrostu informacji, współczynnik split
    """
    przyrost = oblicz_przyrost_informacji(df, kolumna)
    wspolczynnik_split = oblicz_wspolczynnik_split(df, kolumna)
    return (przyrost / wspolczynnik_split if wspolczynnik_split != 0 else 0, wspolczynnik_split)

if __name__ == "__main__":
    # Wczytaj dane z pliku
    plik_kategorii = "gielda.txt"
    df_kategorie = pd.read_csv(plik_kategorii, sep=",", header=None)

    # Oblicz i wyświetl zrównoważony przyrost informacji oraz współczynnik split dla każdej kolumny oprócz ostatniej
    for col in df_kategorie.columns[:-1]:
        gain_ratio, split_info = oblicz_zrownowazony_przyrost(df_kategorie, col)
        print(f"Współczynnik split dla atrybutu {col}: {split_info}")
        print(f"Zrównoważony przyrost informacji dla atrybutu {col}: {gain_ratio}")

# Zrównoważony przyrost informacji (Gain Ratio) to miara używana w drzewach decyzyjnych,
# która uwzględnia zarówno przyrost informacji, jak i liczbę możliwych wartości atrybutu.
# Jest to stosunek przyrostu informacji do współczynnika split, co pomaga zredukować
# uprzedzenia wobec atrybutów z wieloma wartościami.
