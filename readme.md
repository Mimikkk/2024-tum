# Projekt zaliczeniowy z przedmiotu "Teoria uczenia maszynowego"

## Temat:

Wpływ dodawania szumu na trafność algorytmów uczenia: np. dodanie dodatkowych cech wejściowych
z czystym szumem (nawet wielu), zaszumienie istniejących cech, zaszumienie wartości na wyjściu (np.
losowe zmiany etykiet klas w części obiektów).

## Opis:

Celem projektu jest zbadanie wpływu dodawania szumu na trafność algorytmów uczenia maszynowego.
Szum może być dodawany na różnych etapach procesu uczenia, np.:

- dodanie dodatkowych cech wejściowych z czystym szumem (nawet wielu),
- zaszumienie istniejących cech,
- zaszumienie wartości na wyjściu (np. losowe zmiany etykiet klas w części obiektów).

## Zadania:

1. Wybór zbioru danych
    - Wybrano zbiór danych "Iris" dostępny w bibliotece sklearn.datasets
    - Oraz zbiór danych "Wine" dostępny w bibliotece sklearn.datasets
2. Implementacja algorytmów uczenia maszynowego
    - Wykorzystano algorytmy: SVM, KNN, Random Forest, MLP oraz prostą sieć neuronową
    - Wykorzystano gotowe implementacje z biblioteki sklearn oraz keras
    - Zaimplementowano funkcję, która zwraca trafność klasyfikacji dla zbioru danych
3. Implementacja sposobów dodawania szumu
    - Dodawanie szumu nowych cech wejściowych z czystym szumem
    - Dodawanie szumu nowych cech wejściowych ze skorelowanym szumem
    - Dodawanie szumu do istniejących cech wejściowych z czystym szumem
    - Dodawanie szumu do istniejących cech wejściowych ze skorelowanym szumem
    - Dodawanie szumu na wyjściu (losowe zmiany etykiet klas w części obiektów)
4. Zbadanie wpływu dodawania szumu na trafność algorytmów uczenia maszynowego
    - Zaimplementowano funkcję, która dodaje szum do zbioru danych
    - Zbadano wpływ dodawania szumu na trafność klasyfikacji dla różnych algorytmów uczenia maszynowego
4. Przeprowadzenie eksperymentów
5. Wyniki i ich analiza
6. Wnioski
