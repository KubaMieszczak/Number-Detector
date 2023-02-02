import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage import color
import os
from tensorflow.keras.models import load_model


#C:\Users\ADMIN\Desktop\Number Detector\number_detector.py
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["SM_FRAMEWORK"] = "tf.keras"
model=load_model('C:/Users/ADMIN/jozponawanie_liczb.h5')

# ustawienie ścieżki do foldera
path = "C:/Users/ADMIN/Desktop/dokumenty/Programming-main/Number Detector/Cyfry"

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
    return (suma_ost,maxy)

def menu(img=0):
    clear()
    print('#' * 100)
    print('    1. Wyjaśnienie działania programu')
    print('    2. Wczytaj obraz')
    print('    3. Start')
    print('    4. Wyświetl wybrany obraz')
    print('    5. Wyjdź')
    print('#' * 100)
    x = int(input('    '))
    if x == 1:
        clear()
        print(
            'Wczytaj z wybranego foldera (Cyfry) obraz o rozmiarze 28x28, następnie wybierz "Start" i zobacz czy \nprogram przewidział twoją liczbę\n\n '
            '   Folder Cyfry powinien znajdować się pod ścieżką ' + path + '\n\n')
        q = input('Wpisz "q" żeby wyjść ')
        if q == 'q':
            clear()
            menu()
    if x == 2:
        clear()
        images = os.listdir(path)
        # wyświetlenie dostępnych plików
        print('Lista dostępnych plików: ')
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
        img1 = color.rgb2gray(img)
        img1 = np.array((img1 * 255).reshape(784, ))
        print('Przewidziana liczba to: {} na {}%'.format(detect(img1)[0],detect(img1)[1]*100))
        # pierwsze funkcja która przerobi odebrany obraz na odpowiednie rozmiary i przewidzi wynik
        # potem funckcja która pokaże jaka to liczba
        q = input('Wpisz "q" żeby wrócić do menu: ')
        if q == 'q':
            menu(img=img)
    if x == 4:
        plt.imshow(img)
        plt.show()
        menu(img=img)
    if x == 5:
        clear()
menu()