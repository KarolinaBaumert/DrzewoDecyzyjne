import pandas as pd
import numpy as np


def oblicz_info(df, kolumna):
    """
    Oblicza wartość funkcji informacji dla podanego atrybutu w tabeli decyzyjnej.
    :param df: DataFrame zawierający dane
    :param kolumna: Nazwa kolumny, dla której obliczamy informację
    :return: Wartość funkcji informacji
    """
    total_rows = len(df)
    wartosci_atrybutu = df[kolumna].unique()
    info = 0

    for wartosc in wartosci_atrybutu:
        podzbior = df[df[kolumna] == wartosc]
        klasy_dec = podzbior.iloc[:, -1].value_counts().to_dict()
        entropia_podzbioru = -sum(
            (count / len(podzbior)) * np.log2(count / len(podzbior)) for count in klasy_dec.values() if count > 0)
        info += (len(podzbior) / total_rows) * entropia_podzbioru

    return info


if __name__ == "__main__":
    plik_kategorii = "gielda.txt"
    df_kategorie = pd.read_csv(plik_kategorii, sep=",", header=None)

    for col in df_kategorie.columns[:-1]:
        info_value = oblicz_info(df_kategorie, col)
        print(f"Informacja dla atrybutu {col}: {info_value}")