import pandas as pd


def wczytaj_tabele(plik, separator=",", naglowki=None):
    """
    Wczytuje dane z pliku tekstowego do DataFrame.
    :param plik: Ścieżka do pliku
    :param separator: Separator danych (domyślnie przecinek)
    :param naglowki: Lista nagłówków kolumn (opcjonalnie)
    :return: DataFrame z wczytanymi danymi
    """
    return pd.read_csv(plik, sep=separator, names=naglowki, header=None)


def analiza_atrybutow(df):
    """
    Analizuje dane: liczy unikalne wartości i ich wystąpienia w każdej kolumnie.
    :param df: DataFrame z danymi
    :return: Słowniki unikalnych wartości i liczby wystąpień
    """
    unikalne_wartosci = {col: df[col].nunique() for col in df.columns[:-1]}
    wystapienia = {col: df[col].value_counts().to_dict() for col in df.columns[:-1]}
    return unikalne_wartosci, wystapienia


if __name__ == "__main__":
    # Wczytanie tabeli z danymi kategorycznymi
    plik_kategorii = "gielda.txt"
    df_kategorie = wczytaj_tabele(plik_kategorii, separator=",")

    # Wczytanie tabeli z danymi liczbowymi
    plik_danych = "gieldaliczby.txt"
    df_liczby = wczytaj_tabele(plik_danych, separator=",")

    # Analiza atrybutów dla obu tabel
    unikalne_kategorie, wystapienia_kategorie = analiza_atrybutow(df_kategorie)
    unikalne_liczby, wystapienia_liczby = analiza_atrybutow(df_liczby)

    # Wyświetlenie wyników
    print("Tabela decyzyjna (kategorie):")
    print(df_kategorie)
    print("\nTabela decyzyjna (liczby):")
    print(df_liczby)

    print("\nMożliwa liczba wartości każdego atrybutu (kategorie):")
    for col, count in unikalne_kategorie.items():
        print(f"Atrybut {col}: {count} unikalnych wartości")

    print("\nMożliwa liczba wartości każdego atrybutu (liczby):")
    for col, count in unikalne_liczby.items():
        print(f"Atrybut {col}: {count} unikalnych wartości")

    print("\nLiczba wystąpień każdej wartości w każdym atrybucie (kategorie):")
    for col, counts in wystapienia_kategorie.items():
        print(f"Atrybut {col}: {counts}")

    print("\nLiczba wystąpień każdej wartości w każdym atrybucie (liczby):")
    for col, counts in wystapienia_liczby.items():
        print(f"Atrybut {col}: {counts}")
