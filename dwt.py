import numpy as np
import pywt
import cv2
from matplotlib import pyplot as plt


def RMSE(I0, I1):
    """ Mesure de la RMSE entre les images I0 et I1 """
    assert I0.shape == I1.shape
    H, W = I0.shape
    return np.sqrt(np.sum((I1.astype(np.float32) - I0.astype(np.float32))**2) / (H * W))


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


def main():
    I = cv2.imread("Images/04.png", 0)
    coeffs_DWT = pywt.wavedec2(I, onde, 'symmetric', level)
    coeff_arr, coeff_slices = pywt.coeffs_to_array(coeffs_DWT)

    # Quantification
    coeff_arr_q = coeff_arr.copy()
    coeff_slices_q = coeff_slices.copy()

    # Quantification
    C = coeff_arr_q[coeff_slices_q[0]]  # Matrice de coefficient à quantifier
    coeff_arr_q[coeff_slices_q[0]] = np.sign(C) * np.fix(np.abs(C) / delta[0])

    for i in range(1, level + 1):
        for key in ['ad', 'dd', 'da']:
            C = coeff_arr_q[coeff_slices_q[i][key]]  # Matrice de coefficient à quantifier
            coeff_arr_q[coeff_slices_q[i][key]] = np.sign(C) * np.fix(np.abs(C) / delta[i])

    # Dequantification
    coeff_arr_rec = coeff_arr_q.copy()
    coeff_slices_rec = reconstruct_coeff_slices(coeff_arr_rec)

    r = 0.5  # A fixer pour minimiser la RMSE

    Q = coeff_arr_rec[coeff_slices_rec[0]]  # Matrice de coefficient à déquantifier
    coeff_arr_rec[coeff_slices_rec[0]] = (Q + r * np.sign(Q)) * delta[0]

    for i in range(1, level + 1):
        for key in ['ad', 'dd', 'da']:
            Q = coeff_arr_rec[coeff_slices_rec[i][key]]  # Matrice de coefficient à déquantifier
            coeff_arr_rec[coeff_slices_rec[i][key]] = (Q + r * np.sign(Q)) * delta[i]

    # Reconstruction
    coeffs_rec = pywt.array_to_coeffs(coeff_arr_rec, coeff_slices_rec)

    # Mise au même format que coeffs_DWT
    coeffs_rec = [coeffs_rec[0]] + [list(coeffs_rec[i].values()) for i in range(1, level + 1)]
    for i in range(1, level + 1):
        coeffs_rec[i] = [coeffs_rec[i][1], coeffs_rec[i][0], coeffs_rec[i][2]]

    I_rec = pywt.waverec2(coeffs_rec, onde)

    diff = np.abs(I_rec.astype(np.int16) - I.astype(np.int16)).astype(np.uint8)

    plt.figure()

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

    plt.show()


if __name__ == "__main__":
    onde = "haar"
    delta = {
        0: 1,
        1: 5,
        2: 10,
        3: 100
    }
    level = len(delta) - 1
    main()
