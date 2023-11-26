# Exactemet 15 de compression
# 

import cv2
import numpy as np
import matplotlib.pyplot as plt


Z = np.array(
    [
        [16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99]
    ]
)


def RMSE(I0, I1):
    assert I0.shape == I1.shape
    H, W = I0.shape
    return np.linalg.norm(I1 - I0) / np.sqrt(H * W)


def zigzag(H, W):
    """ Retourne une liste de tuples représentant les coordonnées du zigzag d'encodage """
    x = 0
    y = 0
    L = [(x,y)]

    b = True

    while x < W and y < H:
        if x == 0 or y == 0 or x == W - 1 or y == H - 1:
            if b and y==0 or not b and y == H-1:
                x += 1
            else:
                y += 1
            b = not b
            L.append((x,y))

        if b:
            x += 1
            y -= 1
        else:
            x -= 1
            y += 1
        L.append((x,y))
    return L[:-1]


def plage(L):
    """ Retourne le codage en plage de la liste """
    i = 1
    n = 0
    res = []
    for k in L:
        if k == i:
            n += 1
        else:
            res.append(n)
            n = 1
            i = 1 - i
    res.append(n)
    return res


def un_plage(L):
    """ Retourne le décodage en plage de la liste """
    res = []
    i = 1
    for n in L:
        res += n * [i]
        i = 1-i
    return res


def calc_size(non_zero_val_matrix, non_zero_pos_matrix):
    non_zero_val = []
    non_zero_val_bits = []
    for non_zero_val_list in non_zero_val_matrix.flatten():
        non_zero_val += non_zero_val_list
        if non_zero_val_list:
            non_zero_val_bits.append(int(np.log(max(np.abs(np.array(non_zero_val_list)))) / np.log(2)) + 1)

    non_zero_pos = []
    for non_zero_pos_list in non_zero_pos_matrix.flatten():
        non_zero_pos += non_zero_pos_list

    N = len(non_zero_pos) + sum(non_zero_val_bits)
    return N


def image_reconstruction(DCT_q):
    """ Reconstruit une image à partir d'une DCT quantifiée """
    I = np.zeros(DCT_q.shape)
    H, W = I.shape

    for i in range(int(H / 8)):
        for j in range(int(W / 8)):
            I[8*i:8*(i+1), 8*j:8*(j+1)] = DCT_q[8*i:8*(i+1), 8*j:8*(j+1)] * Z
    I = cv2.dct(I, flags=cv2.DCT_INVERSE)
    return I


def DCT_reconstruction(val, L):
    H,W = val.shape  # val et L sont de taille (H/8, W/8) avec H,W la taille de l'image de base
    H *= 8
    W *= 8
    DCT = np.zeros((H,W))
    zz_coords = zigzag(8,8)

    for i in range(int(H / 8)):
        for j in range(int(W / 8)):
            l = un_plage(L[i,j])
            v = val[i,j]
            n = 0
            for k, coord in enumerate(zz_coords):
                if l[k]:
                    DCT[coord[1] + 8*i, coord[0] + 8*j] = v[n]
                    n += 1
    return DCT


def compression(DCT):
    global Z

    H, W = DCT.shape
    T = DCT.copy()

    for i in range(0, H, 8):
        for j in range(0, W, 8):
            T[i:i+8, j:j+8] /= Z

    T = np.round(T)

    non_zero_val_matrix = []  # Matrice H/8 x W/8 contenant les listes de valeurs qtifiees non nulles
    non_zero_pos_matrix = []  # Matrice H/8 x W/8 contenant les listes de 0 et 1 indiquant les positions des valeurs non nulles
    zz_coords = zigzag(8,8)

    for i in range(0, H, 8):
        line_non_zero_val = []
        line_non_zero_pos = []
        for j in range(0, W, 8):
            block_non_zero_pos = []
            block_non_zero_val = []
            for coord in zz_coords:
                x = j + coord[0]
                y = i + coord[1]
                non_zero = int(T[y, x] != 0)
                block_non_zero_pos.append(non_zero)
                if non_zero:
                    block_non_zero_val.append(T[y, x])
            line_non_zero_pos.append(plage(block_non_zero_pos))
            line_non_zero_val.append(block_non_zero_val)
        non_zero_pos_matrix.append(line_non_zero_pos)
        non_zero_val_matrix.append(line_non_zero_val)

    non_zero_pos_matrix = np.array(non_zero_pos_matrix, dtype=object)
    non_zero_val_matrix = np.array(non_zero_val_matrix, dtype=object)

    size = calc_size(non_zero_val_matrix, non_zero_pos_matrix)
    print(f"Taux de compression : {H * W * 8 / size}")

    DCT_rec = DCT_reconstruction(non_zero_val_matrix, non_zero_pos_matrix)

    return image_reconstruction(DCT_rec)


def main():
    I = cv2.imread("Images/04.png", 0)
    I_autocorr = np.corrcoef(I)

    plt.figure(1)
    plt.subplot(1,2,1)
    plt.imshow(I, 'gray')
    plt.title('Image')
    plt.subplot(1,2,2)
    plt.imshow(I_autocorr, 'gray')
    plt.title('Autocorrelation')
    plt.show()

    DCT = cv2.dct(I.astype(np.float32))
    DCTlog = np.log(np.abs(DCT))
    DCT_autocorr = np.corrcoef(DCT)

    plt.figure(2)
    plt.subplot(1,3,1)
    plt.imshow(DCT, 'gray')
    plt.title('DCT')
    plt.subplot(1,3,2)
    plt.imshow(DCTlog, 'gray')
    plt.title('log(|DCT|)')
    plt.subplot(1,3,3)
    plt.imshow(DCT_autocorr, 'gray')
    plt.title('Autocorrelation de la DCT')
    plt.show()

    I_rec = compression(DCT)
    plt.figure(3)
    plt.subplot(1,2,1)
    plt.imshow(I, "gray")
    plt.title('Image originale')
    plt.subplot(1,2,2)
    plt.imshow(I_rec, "gray")
    plt.title('Image reconstruite')
    plt.show()

    print(f"RMSE : {RMSE(I, I_rec)}")


if __name__ == "__main__":
    main()