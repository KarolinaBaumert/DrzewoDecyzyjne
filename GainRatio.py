import pandas as pd
import numpy as np
from InfoXT import oblicz_info
from Gain import oblicz_przyrost_informacji


def oblicz_zrownowazony_przyrost(df, kolumna):
    """
    Oblicza zrównoważony przyrost informacji dla podanego atrybutu.
    :param df: DataFrame zawierający dane
    :param kolumna: Nazwa kolumny, dla której obliczamy zrównoważony przyrost informacji
    :return: Wartość zrównoważonego przyrostu informacji
    """
    przyrost = oblicz_przyrost_informacji(df, kolumna)
    wartosci_atrybutu = df[kolumna].value_counts(normalize=True).values
    wspolczynnik_split = -sum(p * np.log2(p) for p in wartosci_atrybutu)

    return przyrost / wspolczynnik_split if wspolczynnik_split != 0 else 0


if __name__ == "__main__":
    # Wczytanie danych
    plik_kategorii = "gielda.txt"
    df_kategorie = pd.read_csv(plik_kategorii, sep=",", header=None)

    # Obliczenie zrównoważonego przyrostu informacji dla każdej kolumny (bez ostatniej decyzyjnej)
    for col in df_kategorie.columns[:-1]:
        gain_ratio = oblicz_zrownowazony_przyrost(df_kategorie, col)
        print(f"Zrównoważony przyrost informacji dla atrybutu {col}: {gain_ratio}")