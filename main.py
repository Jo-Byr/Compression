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

encode_diff = True


def RMSE(I0, I1):
    assert I0.shape == I1.shape
    H, W = I0.shape
    return np.sqrt(np.sum((I1 - I0)**2)) / np.sqrt(H * W)


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


def bits(i):
    return int(np.log(i) / np.log(2)) + 1


def calc_size(*args):
    tot = 0
    for arg in args:
        signed = False
        blocks_bits = []
        for block_list in arg:
            L = np.array(block_list)
            if np.any(L):
                necessary_bits = bits(max(np.abs(L)))
                blocks_bits.append(len(block_list) * necessary_bits)
                if not signed and np.any(L < 0):
                    signed = True
        tot += np.sum(blocks_bits) + len(np.sum(arg)) * int(signed)
    return tot


def image_reconstruction(DCT_q):
    """ Reconstruit une image à partir d'une DCT quantifiée """
    I_rec = np.zeros(DCT_q.shape)
    H, W = I_rec.shape

    for i in range(0, H, 8):
        for j in range(0, W, 8):
            I_rec[i:i + 8, j:j+8] = cv2.idct(DCT_q[i:i + 8, j:j+8] * Z)
    I_rec = np.clip(I_rec, 0, 255).astype(np.uint8)
    return I_rec


def DCT_reconstruction(val, L):
    H, W = val.shape  # val et L sont de taille (H/8, W/8) avec H,W la taille de l'image de base
    H *= 8
    W *= 8
    DCT = np.zeros((H, W))
    zz_coords = zigzag(8, 8)

    for i in range(int(H/8)):
        for j in range(int(W/8)):
            l = un_plage(L[i, j])
            v = val[i, j]
            assert len(np.where(np.array(l) == 1)[0]) == len(v)
            n = 0
            top_left = True
            for k, coord in enumerate(zz_coords):
                if encode_diff:
                    value = 0
                    if l[k]:
                        value = v[n]
                        if top_left:
                            if not(i == 0 and j == 0):
                                value += last_top_left_val
                            last_top_left_val = value
                        n += 1
                    elif k == 0:
                        value = last_top_left_val
                    DCT[coord[1] + 8*i, coord[0] + 8*j] = value
                    top_left = False
                else:
                    if l[k]:
                        DCT[coord[1] + 8 * i, coord[0] + 8 * j] = v[n]
                        n += 1
    return DCT


def compression(I):
    global Z
    H, W = I.shape
    If = I.astype(np.float32)

    non_zero_val_matrix = []  # Matrice H/8 x W/8 contenant les listes de valeurs qtifiees non nulles
    non_zero_pos_matrix = []  # Matrice H/8 x W/8 contenant les listes de 0 et 1 indiquant les positions des valeurs non nulles
    zz_coords = zigzag(8, 8)

    for i in range(0, H, 8):
        line_non_zero_val = []
        line_non_zero_pos = []
        for j in range(0, W, 8):
            dct = cv2.dct(If[i:i+8, j:j+8])
            T = np.fix(dct/Z)

            block_non_zero_pos = []
            block_non_zero_val = []
            for coord in zz_coords:
                x = coord[0]
                y = coord[1]
                value = T[y, x]
                if encode_diff:
                    if x == y == 0:
                        if not(i == j == 0):
                            value -= last_top_left_val
                        last_top_left_val = T[y, x]

                non_zero = int(value != 0)
                block_non_zero_pos.append(non_zero)

                if non_zero:
                    block_non_zero_val.append(value)

            line_non_zero_pos.append(plage(block_non_zero_pos))
            line_non_zero_val.append(block_non_zero_val)

        non_zero_pos_matrix.append(line_non_zero_pos)
        non_zero_val_matrix.append(line_non_zero_val)

    non_zero_pos_matrix = np.array(non_zero_pos_matrix, dtype=object)
    non_zero_val_matrix = np.array(non_zero_val_matrix, dtype=object)

    size = calc_size(non_zero_val_matrix.flatten(), non_zero_pos_matrix.flatten())
    taux_comp = H * W * 8 / size

    DCT_rec = DCT_reconstruction(non_zero_val_matrix, non_zero_pos_matrix)

    return image_reconstruction(DCT_rec), taux_comp


def main():
    I = cv2.imread("Images/04.png", 0)
    I_autocorr = np.corrcoef(I)

    plt.figure()

    plt.subplot(1, 3, 1)
    plt.imshow(I, 'gray')
    plt.title('Image')

    plt.subplot(1, 3, 2)
    plt.imshow(I_autocorr, 'gray')
    plt.title('Autocorrelation')

    plt.subplot(1, 3, 3)
    plt.hist(I.ravel(), 256, [0, 256])
    plt.title('Histogramme de distribution des intensités')

    plt.show()

    DCT = cv2.dct(I.astype(np.float32))
    DCTlog = np.log(np.abs(DCT))
    DCT_autocorr = np.corrcoef(DCT)

    plt.figure(2)

    plt.subplot(2, 2, 1)
    plt.imshow(DCT, 'gray')
    plt.title('DCT')

    plt.subplot(2, 2, 2)
    plt.imshow(DCTlog, 'gray')
    plt.title('log(|DCT|)')

    plt.subplot(2, 2, 3)
    plt.imshow(DCT_autocorr, 'gray')
    plt.title('Autocorrelation de la DCT')

    plt.subplot(2, 2, 4)
    plt.hist(DCT.ravel(), bins=500, range=[-200, 200])
    #plt.hist(DCTlog.ravel(), bins=500)
    plt.title('Histogramme de distribution des intensités')

    plt.show()

    I_rec, taux_comp = compression(I)
    cv2.imwrite("res2.png", I_rec)

    print(f"RMSE : {RMSE(I, I_rec)}")
    print(f"Taux de compression : {taux_comp}")

    plt.figure(3)
    plt.subplot(1,3,1)
    plt.imshow(I, "gray")
    plt.title('Image originale')
    plt.subplot(1,3,2)
    plt.imshow(I_rec, "gray")
    plt.title('Image reconstruite')
    plt.subplot(1,3,3)
    diff = np.abs(I_rec.astype(np.int16) - I.astype(np.int16))
    plt.imshow(diff, "gray")
    plt.title('Différence')
    plt.show()


def test_rand():
    global Z
    I = cv2.imread("Images/04.png", 0)
    I_rec, taux_comp = compression(I)
    best = taux_comp

    Z_i = Z
    best_Z = Z_i

    n = 5
    for i in range(100):
        r = np.fix(np.random.rand(8, 8) * 2 * (n + 1) - (n + 1))
        Z = Z_i + r
        I_rec, taux_comp = compression(I)
        err = RMSE(I, I_rec)
        if taux_comp > best:
            best = taux_comp
            best_Z = Z
            print(best, err)
        print(i/100)
    print(best)
    print(best_Z)


if __name__ == "__main__":
    #Z = np.ones((8,8))
    #Z = np.round(Z.astype(np.float32) * 2.0).astype(np.uint16)
    main()
    #test_rand()
