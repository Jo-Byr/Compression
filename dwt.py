import numpy as np
import pywt
import cv2
from matplotlib import pyplot as plt
from random import random

DCT_keys = ['ad', 'dd', 'da']


def RMSE(I0, I1):
    """ Mesure de la RMSE entre les images I0 et I1 """
    assert I0.shape == I1.shape
    H, W = I0.shape
    return np.sqrt(np.sum((I1.astype(np.float32) - I0.astype(np.float32))**2) / (H * W))


def bits(i):
    """ Retourne le nombre de bits nécessaire pour encoder l'entier i """
    if i == 0:
        return 1
    return int(np.log(i) / np.log(2)) + 1


def reconstruct_coeff_slices(coeff_arr):
    """ Reconstruit la liste de dictionnaire coeff_slices associée aux coefficients coeff_arr """
    H, W = coeff_arr.shape
    assert H == W  # On ne supporte que les images carrées
    size0 = int(H / 2**level)
    res = [(slice(None, size0, None), slice(None, size0, None))]
    for i in range(1, level + 1):
        size1 = int(H / 2**(level - i + 1))
        size2 = int(H / 2**(level - i))
        res.append({'ad': (slice(None, size1, None), slice(size1, size2, None)),
                    'da': (slice(size1, size2, None), slice(None, size1, None)),
                    'dd': (slice(size1, size2, None), slice(size1, size2, None))})
    return res


def size_DWT(coeff_arr, coeff_slices):
    """ Retourne le nombre de bits nécessaire pour encoder les coefficients de DWT donnés sans compression """
    tot_bits = 0

    # Le 1er bloc est complétement positif donc il n'y a pas besoin de bit de signe
    block0 = coeff_arr[coeff_slices[0]]
    bits_0 = np.size(block0) * bits(np.amax(block0))  # Nombre de bits pour encoder le 1er bloc
    bits_0 += 3  # 3 bits pour passer le nombre de bits pour encoder un nombre

    tot_bits += bits_0
    for i in range(1, level + 1):
        for key in DCT_keys:
            # Les blocs secondaires ont des valeurs négatives, il faut donc un bit de signe
            block = coeff_arr[coeff_slices[i][key]]
            bits_block = np.size(block) * (1 + bits(np.amax(np.abs(block))))  # Nombre de bits pour encoder le bloc
            bits_block += 3  # 3 bits octet pour passer le nombre de bits pour encoder un nombre
            tot_bits += bits_block
    return tot_bits


def compressed_block_bit_size(plage_encoded_coeff, signed):
    """ Retourne le nombre de bits nécessaire pour encoder la liste de coefficients, encodés en plage, passée en
    argument """
    # Nb de bits nécessaire pour coder les tailles de plage
    size_bits = bits(np.max(np.array([c[1] for c in plage_encoded_coeff])))
    # Liste des valeurs prises par la liste
    values = np.array([c[0] for c in plage_encoded_coeff])
    # Nb de bits nécessaire pour coder les valeurs de plage
    val_bits = bits(np.max(np.abs(values)))
    if signed:
        val_bits += 1  # +1 pour le signe

    overhead = 2 * 8  # Ajout de 2 octets pour passer les valeurs de size_bits et val_bits
    return overhead + len(plage_encoded_coeff) * (size_bits + val_bits)


def calc_taux_comp(compressed_coeff_arr, H, W):
    """ Retourne le taux de compression obtenu pour la liste de coefficients compressés en plage donnée """
    total_size = 0

    # Taille en bits du bloc principal
    C = compressed_coeff_arr[0]
    total_size += compressed_block_bit_size(C, signed=False)  # Le bloc d'approximation n'a pas de valeurs négatives

    # Itération sur les levels
    for i in range(1, level + 1):
        for j in range(len(DCT_keys)):
            C = compressed_coeff_arr[i][j]
            total_size += compressed_block_bit_size(C, signed=True)

    # Bits pour encoder delta:
    # 3 pour passer le nombre de bits sur lesquels les valeurs sont encodées
    total_size += 3 + len(delta) * bits(max(delta.values()))  # Bits pour encoder delta
    return H * W * 8 / total_size


def plage(mat):
    """ Encode en plage la matrice donnée """
    M = mat.flatten()
    assert len(M) > 0
    L = []
    count = 0
    val = M[0]
    for i in M:
        if i == val:
            count += 1
        else:
            L.append((val, count))
            val = i
            count = 1
    L.append((val, count))
    return L


def unplage(L):
    """ Retourne une liste correspondant au décodage en plage de la liste L donnée """
    M = []
    for val, count in L:
        M += count * [val]
    return np.array(M)


def quantification(I):
    """ Retourne les coefficients de DWT quantifiés de l'image donnée """
    coeffs_DWT = pywt.wavedec2(I, onde, 'symmetric', level)
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs_DWT)

    # Quantification
    coeff_arr_q = coeff_arr.copy()
    coeff_slices_q = coeff_slices.copy()

    # Quantification
    C = coeff_arr_q[coeff_slices_q[0]]  # Matrice de coefficient à quantifier
    coeff_arr_q[coeff_slices_q[0]] = np.sign(C) * np.fix(np.abs(C) / delta[0])

    for i in range(1, level + 1):
        for key in DCT_keys:
            C = coeff_arr_q[coeff_slices_q[i][key]]  # Matrice de coefficient à quantifier
            coeff_arr_q[coeff_slices_q[i][key]] = np.sign(C) * np.fix(np.abs(C) / delta[i])
    return coeff_arr_q, coeff_slices_q


def compression(coeff_arr, coeff_slices):
    """ Retourn les coefficients de DCT donnés codés en plage """
    compressed_coeff_arr = []
    C = coeff_arr[coeff_slices[0]]
    compressed_coeff_arr.append(plage(C))

    for i in range(1, level + 1):
        tup = []
        for key in DCT_keys:
            C = coeff_arr[coeff_slices[i][key]]
            tup.append(plage(C))
        compressed_coeff_arr.append(tuple(tup))
    return compressed_coeff_arr


def decompression(compressed_coeff_arr, coeff_slices):
    """ Retourne la matrice de coefficients DWT correspondant aux coefficients en plage dans compressed_coeff_arr """
    C = unplage(compressed_coeff_arr[0])

    # Récupération des dimensions de l'image
    l = np.sqrt(len(C))
    assert int(l) == l  # On ne traite que des images carrées
    H = 2**level * int(l)
    W = H

    coeff_arr = np.zeros((H, W))
    size0 = int(H / 2**level)
    coeff_arr[coeff_slices[0]] = C.reshape((size0, size0))

    for i in range(1, level + 1):
        for j, key in enumerate(DCT_keys):
            C = unplage(compressed_coeff_arr[i][j])
            size = int(H / 2**(level - i + 1))
            coeff_arr[coeff_slices[i][key]] = C.reshape((size, size))

    return coeff_arr


def reconstruction(compressed_coeff_arr, coeff_slices_rec):
    coeff_arr_rec = decompression(compressed_coeff_arr, coeff_slices_rec)

    r = 0.35  # A fixer pour minimiser la RMSE

    Q = coeff_arr_rec[coeff_slices_rec[0]]  # Matrice de coefficient à déquantifier
    coeff_arr_rec[coeff_slices_rec[0]] = (Q + r * np.sign(Q)) * delta[0]

    for i in range(1, level + 1):
        for key in DCT_keys:
            Q = coeff_arr_rec[coeff_slices_rec[i][key]]  # Matrice de coefficient à déquantifier
            coeff_arr_rec[coeff_slices_rec[i][key]] = (Q + r * np.sign(Q)) * delta[i]

    # Reconstruction
    coeffs_rec = pywt.array_to_coeffs(coeff_arr_rec, coeff_slices_rec)

    # Mise au même format que coeffs_DWT
    coeffs_rec = [coeffs_rec[0]] + [list(coeffs_rec[i].values()) for i in range(1, level + 1)]
    for i in range(1, level + 1):
        coeffs_rec[i] = [coeffs_rec[i][1], coeffs_rec[i][0], coeffs_rec[i][2]]

    I_rec = pywt.waverec2(coeffs_rec, onde)
    return np.clip(I_rec, 0, 255).astype(np.uint8)


def main():
    I = cv2.imread("Images/04.png", 0)
    H, W = I.shape

    coeff_arr_q, coeff_slices_q = quantification(I)

    coeff_slices_rec = reconstruct_coeff_slices(coeff_arr_q)  # Reconstruits pour signifier qu'on a pas besoin de les passer
    compressed_coeff_arr = compression(coeff_arr_q, coeff_slices_rec)

    taux_comp = calc_taux_comp(compressed_coeff_arr, H, W)

    I_rec = reconstruction(compressed_coeff_arr, coeff_slices_rec)

    diff = np.abs(I_rec.astype(np.int16) - I.astype(np.int16)).astype(np.uint8)

    coeffs_DWT = pywt.wavedec2(I, onde, 'symmetric', level)
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs_DWT)

    plt.figure(1)
    plt.subplot(1, 3, 1)
    plt.imshow(I, 'gray')
    plt.title('Image de base')
    plt.subplot(1, 3, 2)
    plt.imshow(coeff_arr, 'gray')
    plt.title('Décomposition en ondelette')
    plt.subplot(1, 3, 3)
    plt.imshow(np.log(1 + np.abs(coeff_arr_q)), 'gray')
    plt.title('Décomposition en ondelette quantifiée (log de v. abs)')

    fig2 = plt.figure(2)
    for i in range(1, level + 1):
        echelle = coeffs_DWT[i][1] + coeffs_DWT[i][1] + coeffs_DWT[i][2]

        hist, bins = np.histogram(echelle, bins=250)
        ax = fig2.add_subplot(level, 1, i)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        ax.bar(center, hist, align='center', width=width)
        plt.title(f'Histogramme de DWT - echelle {i}')

    fig2 = plt.figure(3)
    for i in range(1, level + 1):
        echelle = coeff_arr_q[coeff_slices[i]['ad']] + coeff_arr_q[coeff_slices[i]['dd']] + coeff_arr_q[coeff_slices[i]['da']]

        hist, bins = np.histogram(echelle, bins=100)
        ax = fig2.add_subplot(level, 1, i)
        width = 0.7 * (bins[1] - bins[0])
        center = (bins[:-1] + bins[1:]) / 2
        ax.bar(center, hist, align='center', width=width)
        plt.title(f'Histogramme de DWT quantifiée - echelle {i}')

    plt.figure(4)

    plt.subplot(1, 3, 1)
    plt.imshow(I, 'gray')
    plt.title('Image de base')

    plt.subplot(1, 3, 2)
    plt.imshow(I_rec, 'gray')
    plt.title('Image compressée')

    plt.subplot(1, 3, 3)
    plt.imshow(diff, 'gray')
    plt.title('Différence')

    print(f"RMSE : {RMSE(I, I_rec)}")
    print(f"Taux de compression : {taux_comp}")

    cv2.imwrite("res_dwt.png", I_rec)
    cv2.imwrite("diff_dwt.png", diff)
    cv2.imwrite("oreille.png", I[10:190, 170:300])
    cv2.imwrite("oreille_dwt.png", I_rec[10:190, 170:300])

    plt.show()


def test():
    I = cv2.imread("Images/04.png", 0)
    H, W = I.shape
    coeff_arr_q, coeff_slices_q = quantification(I)
    coeff_slices_rec = reconstruct_coeff_slices(coeff_arr_q)  # Reconstruits pour signifier qu'on a pas besoin de les passer
    compressed_coeff_arr = compression(coeff_arr_q, coeff_slices_rec)
    I_rec = reconstruction(compressed_coeff_arr, coeff_slices_rec)

    best_TC = calc_taux_comp(compressed_coeff_arr, H, W)
    best_RMSE = RMSE(I, I_rec)
    n = 5000
    for i in range(n):
        delta[0] = max(1, 10 * random())
        delta[1] = max(delta[0], 50 * random())
        delta[2] = max(delta[1], 200 * random())
        delta[3] = max(delta[2], 500 * random())

        coeff_arr_q, coeff_slices_q = quantification(I)
        coeff_slices_rec = reconstruct_coeff_slices(coeff_arr_q)  # Reconstruits pour signifier qu'on a pas besoin de les passer
        compressed_coeff_arr = compression(coeff_arr_q, coeff_slices_rec)
        I_rec = reconstruction(compressed_coeff_arr, coeff_slices_rec)
        TC = calc_taux_comp(compressed_coeff_arr, H, W)
        err = RMSE(I, I_rec)
        print(i / n)
        if abs(TC - 15) < abs(best_TC - 15) and err < best_RMSE:
            best_TC = TC
            best_RMSE = err
            best_delta = delta
            print(best_TC, best_RMSE)
            print(best_delta)


if __name__ == "__main__":
    onde = "haar"
    delta = {0: 14, 1: 31, 2: 49, 3: 61}
    level = len(delta) - 1
    #test()
    main()
