import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import color
import os
from tensorflow.keras.models import load_model



#model=load_model('rozponawanie_liczb.h5')

clear=lambda:os.system('cls')
def detect(obraz):
    maxy = 0
    wynik = model.predict(np.array([obraz]))
    wynik = list(wynik.reshape(wynik.shape[0] * wynik.shape[1], ))
    for el in wynik:
        if el > maxy:
            maxy = el
    suma = 0
    for elem in wynik:
        if elem != maxy:
            suma = suma + 1
        if elem == maxy:
            suma_ost = suma
    return suma_ost

# def prediction(img):
#     img=color.rgb2gray(img)
#     img=img.reshape(784,)
#     return img
# print(prediction(cv2.imread("C:/Users/ADMIN/Desktop/Cyfry/1.png")).shape)

def menu(img=0):
    print('#' * 100)
    print('    1. Wyjaśnienie działania programu')
    print('    2. Wczytaj obraz')
    print('    3. Start')
    print('    4. Wyświetl wybrany obraz')
    print('    5. Wyjdź')
    print('#' * 100)
    x = int(input('    '))
    if x == 1:
        print(
            'Wczytaj z wybranego foldera (Cyfry) obraz o rozmiarze 28x28, następnie wybierz "Start" i zobacz czy program przewidział twoją liczbę')
        q = input('Wpisz "q" żeby wyjść')
        if q == 'q':
            clear()
            menu()
    if x == 2:
        # ustawienie ścieżki do foldera
        path = "C:/Users/ADMIN/Desktop/Cyfry"
        images = os.listdir(path)
        # wyświetlenie dostępnych plików
        print('Lista dostępnuch plików: ')
        index = 0
        slownik = {}
        for elem in images:
            print('{}. {}'.format(index, elem))
            slownik[index] = elem
            index = index + 1
        # wybranie pliku do detekcji
        x = int(input('Wybierz numer pliku: \n'))
        wybrane = slownik[x]
        print('Wybrany plik: {}\n'.format(wybrane))
        # wczytanie pliku
        img = cv2.imread(path + '/' + wybrane)
        clear()
        menu(img=img)
    if x == 3:
        img = color.rgb2gray(img)
        img = np.array((img * 255).reshape(784, ))
        print('Przewidziana liczba to: {}'.format(detect(img)))
        # pierwsze funkcja która przerobi odebrany obraz na odpowiednie rozmiary i przewidzi wynik
        # potem funckcja która pokaże jaka to liczba
        clear()
        menu(img=img)
    if x == 4:
        plt.imshow(img)
        plt.show()
        q = input('Wpisz "q" żeby wrócić do menu: ')
        if q == 'q':
            menu()
menu()