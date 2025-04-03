import pandas as pd
import numpy as np  # Poprawienie importu numpy

def oblicz_info(df, kolumna):
    """
    Oblicza wartość funkcji informacji dla podanego atrybutu w tabeli decyzyjnej.
    :param df: DataFrame zawierający dane
    :param kolumna: Nazwa kolumny, dla której obliczamy informację
    :return: Wartość funkcji informacji
    """
    total_rows = len(df)  # Całkowita liczba wierszy w DataFrame
    wartosci_atrybutu = df[kolumna].unique()  # Unikalne wartości atrybutu
    info = 0

    for wartosc in wartosci_atrybutu:
        podzbior = df[df[kolumna] == wartosc]  # Podzbiór danych dla danej wartości atrybutu
        klasy_dec = podzbior.iloc[:, -1].value_counts().to_dict()  # Liczba wystąpień każdej klasy decyzyjnej
        entropia_podzbioru = -sum(
            (count / len(podzbior)) * np.log2(count / len(podzbior)) for count in klasy_dec.values() if count > 0)  # Entropia podzbioru
        info += (len(podzbior) / total_rows) * entropia_podzbioru  # Dodanie ważonej entropii podzbioru do całkowitej informacji

    return info  # Zwrócenie wartości funkcji informacji

if __name__ == "__main__":
    plik_kategorii = "gielda.txt"
    try:
        df_kategorie = pd.read_csv(plik_kategorii, sep=",", header=None)  # Wczytanie danych z pliku
    except FileNotFoundError:
        print(f"Plik {plik_kategorii} nie został znaleziony.")
        exit(1)
    except pd.errors.EmptyDataError:
        print(f"Plik {plik_kategorii} jest pusty.")
        exit(1)
    except pd.errors.ParserError:
        print(f"Plik {plik_kategorii} zawiera błędy i nie może zostać wczytany.")
        exit(1)

    for col in df_kategorie.columns[:-1]:
        info_value = oblicz_info(df_kategorie, col)  # Obliczenie informacji dla każdej kolumny
        print(f"Informacja dla atrybutu {col}: {info_value}")  # Wyświetlenie wyniku

# Funkcja informacji mierzy niepewność związaną z atrybutem w zbiorze danych.
# Im wyższa wartość funkcji informacji, tym większa niepewność.
# W kontekście drzew decyzyjnych, funkcja informacji pomaga wybrać atrybuty, które najlepiej dzielą dane.
# Wybieramy atrybuty o najniższej wartości funkcji informacji, ponieważ niższa entropia oznacza mniejszą niepewność co do klasy decyzyjnej po podziale.
# Niepewność związana z atrybutem odnosi się do niepewności co do klasy decyzyjnej, gdy znamy wartość tego atrybutu.
# Wysoka entropia oznacza, że wiedza o wartości atrybutu niewiele mówi nam o klasie decyzyjnej, podczas gdy niska entropia oznacza, że wartość atrybutu daje nam dużo informacji o klasie decyzyjnej.

