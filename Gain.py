import pandas as pd
import numpy as np  # Poprawienie importu numpy
from InfoXT import oblicz_info

def oblicz_przyrost_informacji(df, kolumna):
    """
    Oblicza przyrost informacji dla podanego atrybutu.
    :param df: DataFrame zawierający dane
    :param kolumna: Nazwa kolumny, dla której obliczamy przyrost informacji
    :return: Wartość przyrostu informacji
    """
    # Obliczanie pełnej entropii dla klasy decyzyjnej
    entropia_pelna = -sum((count / len(df)) * np.log2(count / len(df))
                          for count in df.iloc[:, -1].value_counts().to_dict().values())
    # Obliczanie informacji dla atrybutu
    info_atrybut = oblicz_info(df, kolumna)
    # Zwracanie przyrostu informacji
    return entropia_pelna - info_atrybut

if __name__ == "__main__":
    # Wczytywanie danych z pliku
    plik_kategorii = "gielda.txt"
    df_kategorie = pd.read_csv(plik_kategorii, sep=",", header=None)

    # Obliczanie i wyświetlanie przyrostu informacji dla każdej kolumny (oprócz ostatniej)
    for col in df_kategorie.columns[:-1]:
        gain = oblicz_przyrost_informacji(df_kategorie, col)
        print(f"Przyrost informacji dla atrybutu {col}: {gain}")

# Przyrost informacji to miara używana w teorii informacji i uczeniu maszynowym,
# która określa, jak dobrze dany atrybut (cecha) rozdziela dane na klasy decyzyjne.
# Wyższy przyrost informacji oznacza, że atrybut lepiej rozdziela dane.

