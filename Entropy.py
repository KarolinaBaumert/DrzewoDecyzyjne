import pandas as pd
import numpy as np
from LoadData import wczytaj_tabele


def oblicz_entropie(df):
    """
    Oblicza entropię dla atrybutu decyzyjnego (ostatnia kolumna).
    :param df: DataFrame z danymi
    :return: Wartość entropii
    """
    decyzje = df.iloc[:, -1]  # Ostatnia kolumna jako atrybut decyzyjny
    wystapienia = decyzje.value_counts(normalize=True)  # Prawdopodobieństwa wartości
    entropia = -sum(p * np.log2(p) for p in wystapienia)
    return entropia


if __name__ == "__main__":
    # Wczytanie tabeli z danymi liczbowymi
    plik_danych = "gieldaliczby.txt"
    df_liczby = wczytaj_tabele(plik_danych, separator=",")

    # Obliczenie entropii
    entropia_liczbowa = oblicz_entropie(df_liczby)

    print(f"Entropia dla danych liczbowych: {entropia_liczbowa:.4f}")

"""
Entropia to miara nieuporządkowania lub niepewności w zbiorze danych.

Wyobraź sobie, że masz worek z kulkami:
Jeśli wszystkie kulki są tego samego koloru, to wiesz od razu, co wyciągniesz – zero niepewności → entropia = 0.
Jeśli masz po równo różnych kolorów, to nie masz pojęcia, co wyciągniesz – maksymalna niepewność → entropia = 1 
(lub więcej, zależnie od liczby kolorów).
"""
