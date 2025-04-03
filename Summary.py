import pandas as pd
from Entropy import oblicz_entropie
from InfoXT import oblicz_info
from Gain import oblicz_przyrost_informacji
from GainRatio import oblicz_wspolczynnik_split, oblicz_zrownowazony_przyrost
from LoadData import wczytaj_tabele

def wybierz_najlepszy_atrybut(df):
    najlepszy_atrybut = None
    najlepszy_gain_ratio = -1

    for col in df.columns[:-1]:
        gain_ratio = oblicz_zrownowazony_przyrost(df, col)
        if gain_ratio[0] > najlepszy_gain_ratio:
            najlepszy_gain_ratio = gain_ratio[0]
            najlepszy_atrybut = col

    return najlepszy_atrybut, najlepszy_gain_ratio

def main():
    plik_kategorii = "gielda.txt"
    df_kategorie = wczytaj_tabele(plik_kategorii, separator=",")

    # Obliczanie entropii dla klasy decyzyjnej
    entropia_pelna = oblicz_entropie(df_kategorie)
    print(f"Info(T) = {entropia_pelna:.4f} \n")

    for idx, col in enumerate(df_kategorie.columns[:-1], start=1):
        attr_name = f"a{idx}"

        # Obliczanie informacji dla atrybutu
        info_atrybut = oblicz_info(df_kategorie, col)
        print(f"Info({attr_name}, T) = {info_atrybut:.4f}")
    print()  # Dodanie linii przerwy po zbiorze wyników

    for idx, col in enumerate(df_kategorie.columns[:-1], start=1):
        attr_name = f"a{idx}"

        # Obliczanie przyrostu informacji dla atrybutu
        gain = oblicz_przyrost_informacji(df_kategorie, col)
        print(f"Gain({attr_name}, T) = {gain:.4f}")
    print()  # Dodanie linii przerwy po zbiorze wyników

    for idx, col in enumerate(df_kategorie.columns[:-1], start=1):
        attr_name = f"a{idx}"

        # Obliczanie współczynnika split
        split_info = oblicz_wspolczynnik_split(df_kategorie, col)
        print(f"SplitInfo({attr_name}, T) = {split_info:.4f}")

    print()
    for idx, col in enumerate(df_kategorie.columns[:-1], start=1):
        attr_name = f"a{idx}"

        # Obliczanie zrównoważonego przyrostu informacji
        gain_ratio = oblicz_zrownowazony_przyrost(df_kategorie, col)
        print(f"GainRatio({attr_name}, T) = {gain_ratio[0]:.4f}")
    print()  # Dodanie linii przerwy po zbiorze wyników

    najlepszy_atrybut, najlepszy_gain_ratio = wybierz_najlepszy_atrybut(df_kategorie)
    print(f"Najlepszy atrybut: {najlepszy_atrybut} z GainRatio = {najlepszy_gain_ratio:.4f}")

if __name__ == "__main__":
    main()
