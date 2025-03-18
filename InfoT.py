import pandas as pd
import numpy as np
from LoadData import wczytaj_tabele


def oblicz_entropie_klasy(df, atrybut_index):
    """
    Oblicza entropię dla danej kolumny warunkowej w odniesieniu do klasy decyzyjnej.
    :param df: DataFrame z danymi
    :param atrybut_index: Indeks kolumny, dla której obliczamy entropię
    :return: Słownik z wartościami entropii dla każdego atrybutu
    """
    decyzje = df.iloc[:, -1]  # Ostatnia kolumna jako atrybut decyzyjny
    atrybut_values = df.iloc[:, atrybut_index]

    # Słownik do przechowywania wyników
    entropia_wartosci_atrybutu = {}

    # Obliczamy entropię dla każdej unikalnej wartości w danym atrybucie
    for value in atrybut_values.unique():
        # Filtrujemy dane na podstawie wartości atrybutu
        subset = df[atrybut_values == value]
        # Liczymy liczbę obiektów w tej grupie
        liczba_obiektow = len(subset)
        # Liczymy liczbę wystąpień klas decyzyjnych
        wystapienia = subset.iloc[:, -1].value_counts(normalize=True)
        # Obliczamy entropię dla tej grupy
        entropia_grupy = -sum(p * np.log2(p) for p in wystapienia)
        # Dodajemy wynik do słownika
        entropia_wartosci_atrybutu[value] = entropia_grupy

    return entropia_wartosci_atrybutu

def oblicz_entropie_dla_atrybutow(df):
    """
    Oblicza entropię dla każdego atrybutu warunkowego w odniesieniu do klasy decyzyjnej.
    :param df: DataFrame z danymi
    :return: Słownik z entropiami dla każdego atrybutu
    """
    entropie = {}
    for idx, col in enumerate(df.columns[:-1]):  # Ignorujemy ostatnią kolumnę (klasę decyzyjną)
        entropie[col] = oblicz_entropie_klasy(df, idx)
    return entropie


def oblicz_entropie_calosciowa(df, atrybut_index):
    """
    Oblicza entropię całkowitą dla atrybutu (kolumny), ignorując klasy decyzyjne.
    :param df: DataFrame z danymi
    :param atrybut_index: Indeks kolumny, dla której obliczamy entropię
    :return: Wartość entropii całkowitej dla tego atrybutu
    """
    atrybut_values = df.iloc[:, atrybut_index]
    liczba_obiektow = len(df)

    # Liczymy liczbę wystąpień dla każdej unikalnej wartości atrybutu
    wystapienia = atrybut_values.value_counts(normalize=True)

    # Obliczamy entropię całkowitą dla tego atrybutu
    entropia_calosciowa = -sum(p * np.log2(p) for p in wystapienia)

    return entropia_calosciowa

def oblicz_zysk_informacji(df):
    """
    Oblicza zysk informacji dla każdego atrybutu warunkowego na podstawie entropii klasy decyzyjnej.
    :param df: DataFrame z danymi
    :return: Słownik z zyskiem informacji dla każdego atrybutu
    """
    # Obliczamy entropię całkowitą (na podstawie klasy decyzyjnej)
    decyzje = df.iloc[:, -1]
    wystapienia_dec = decyzje.value_counts(normalize=True)
    entropia_decyzji = -sum(p * np.log2(p) for p in wystapienia_dec)

    # Obliczamy zysk informacji dla każdego atrybutu
    zysk_informacji = {}
    for idx, col in enumerate(df.columns[:-1]):
        # Obliczamy entropię całkowitą dla atrybutu
        entropia_calosciowa = oblicz_entropie_calosciowa(df, idx)

        # Obliczamy entropię warunkową
        entropia_wartosci = oblicz_entropie_klasy(df, idx)

        # Zysk informacji = Entropia całkowita - Suma (P(wartość atrybutu) * Entropia dla tej wartości)
        zysk_atrybutu = entropia_calosciowa - sum(
            (len(df[df.iloc[:, idx] == value]) / len(df)) * entropia
            for value, entropia in entropia_wartosci.items()
        )
        zysk_informacji[col] = zysk_atrybutu

    return zysk_informacji

if __name__ == "__main__":
    # Wczytaj dane z pliku gielda.txt
    plik_danych = "gielda.txt"
    df_kategorie = wczytaj_tabele(plik_danych, separator=",")

    # Obliczanie entropii dla wszystkich atrybutów
    entropie_atrybutow = oblicz_entropie_dla_atrybutow(df_kategorie)

    # Obliczanie entropii całkowitej dla każdego atrybutu (kolumny)
    entropie_calosciowe = {}
    for idx, col in enumerate(df_kategorie.columns[:-1]):
        entropie_calosciowe[col] = oblicz_entropie_calosciowa(df_kategorie, idx)

    # Wyświetlanie wyników
    for atrybut, entropie_a in entropie_atrybutow.items():
        print(f"Entropia dla atrybutu {atrybut}:")
        for wartosc, entropia in entropie_a.items():
            print(f"  Wartość '{wartosc}': {entropia:.4f}")

        # Dodajemy entropię całkowitą dla tego atrybutu
        print(f"  Entropia całkowita: {entropie_calosciowe[atrybut]:.4f}")
        print()

    # Obliczanie zysku informacji dla wszystkich atrybutów
    zysk_informacji = oblicz_zysk_informacji(df_kategorie)

    # Wyświetlanie wyników
    for atrybut, zysk in zysk_informacji.items():
        print(f"Zysk informacji dla atrybutu {atrybut}: {zysk:.4f}")
"""
Entropia według klas decyzyjnych (Info(T)) mierzy, jak dobrze dany atrybut (np. old/mid/new) "wyjaśnia" decyzje (up/down).

Weźmy pierwszy atrybut z gielda.txt: old, mid, new:

Rozkład decyzji:
old: Wszystkie 3 wiersze mają decyzję down. Entropia = 0.
mid: Są 4 wiersze: 2 razy down, 2 razy up. Entropia = 1.
new: Są 3 wiersze: 1 raz down, 2 razy up. Entropia ≈ 0.918 (prawa logarytmiczna).

Liczba 10 pochodzi z liczby wierszy w tabeli decyzyjnej.
Proporcje (p(v)):
old: 3/10.
mid: 4/10.
new: 3/10.

Entropia według klas decyzyjnych (Info(T)) jest obliczana dla konkretnej kolumny (tj. jednego atrybutu naraz). W praktyce oznacza to:
Analizujemy wartości w jednej kolumnie i sprawdzamy, jakie wyniki (decyzje) są z nimi powiązane.
Nie uwzględniamy wszystkich kolumn jednocześnie (tj. nie analizujemy konkretnego zestawu jak old, yes, swr).

Znaczenie Info(T) w praktyce
Entropia według klas decyzyjnych (Info(T)) jest podstawą do obliczenia tzw. zysku informacji (Information Gain). 
Zysk informacji mówi nam, jak wiele niepewności w decyzjach udało się zredukować dzięki analizowanemu atrybutowi.
"""