import cv2
import numpy as np
import matplotlib.pyplot as plt

# Matrice de quantification JPEG
Z = np.array(
    [
        [16, 11, 10, 16, 24,  40,  51,  61],
        [12, 12, 14, 19, 26,  58,  60,  55],
        [14, 13, 16, 24, 40,  57,  69,  56],
        [14, 17, 22, 29, 51,  87,  80,  62],
        [18, 22, 37, 56, 68,  109, 103, 77],
        [24, 35, 55, 64, 81,  104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ]
)

# Matrice de quantification alternative permettant un meilleur taux de compression
Z2 = np.array(
    [[16,  12,  12,  18,  26,  40,  50,  56],
     [11,  13,  19,  23,  31,  56,  61,  50],
     [15,  16,  16,  26,  41,  53,  66,  59],
     [17,  19,  19,  30,  46,  82,  84,  61],
     [22,  24,  39,  53,  66, 109, 108,  75],
     [22,  36,  53,  64,  80, 106, 112,  91],
     [51,  64,  79,  87, 106, 120, 115,  97],
     [75,  95,  94, 103, 115, 102, 102,  96]],
)


def RMSE(I0, I1):
    """ Mesure de la RMSE entre les images I0 et I1 """
    assert I0.shape == I1.shape
    H, W = I0.shape
    return np.sqrt(np.sum((I1.astype(np.float32) - I0.astype(np.float32))**2) / (H * W))


def zigzag(H, W):
    """ Retourne une liste de tuples représentant les coordonnées du zigzag d'encodage d'ume image HxW """
    x = 0
    y = 0
    L = [(x, y)]

    b = True

    while x < W and y < H:
        if x == 0 or y == 0 or x == W - 1 or y == H - 1:
            if b and y == 0 or not b and y == H-1:
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
    """ Retourne le codage en plage de la liste L """
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
    """ Retourne le décodage en plage de la liste L """
    res = []
    i = 1
    for n in L:
        res += n * [i]
        i = 1-i
    return res


def bits(i):
    """ Retourne le nombre de bits nécessaire pour encoder l'entier i """
    return int(np.log(i) / np.log(2)) + 1


def pos_encoding(L):
    """ Prend la liste des positions non nulles [1, 1, 0, 0, 1, 0, 0, 0...] encodée en plage et retourne le nombre
     de bits optimal pour encoder cette liste """
    max_bits = bits(max(L))  # Nombre de bits pour encoder le nombre maximal de la liste sur un bloc
    min_len_bits = None
    min_len = None
    for b in range(1, max_bits + 1):
        n = 0
        for val in L:
            v = val
            while v > 0:
                n += b
                v -= 2**b - 1
                if v > 0:
                    n += b
        if min_len is None or n < min_len:
            min_len = n
            min_len_bits = b
    return min_len_bits


def calc_size(non_zero_val_matrix, non_zero_pos_matrix):
    """ Retourne le nombre de bits nécessaire pour encoder l'image compressée
    non_zero_val_matrix est une liste de liste comportant les coefficients non nuls de chaque bloc
    non_zero_pos_matrix est une liste de liste comportant les positions des coefficients non nuls de chaque bloc,
    encodée en plage """
    non_zero_val_bits = []
    for non_zero_val_list in non_zero_val_matrix.flatten():
        if non_zero_val_list:
            # 3 bits pour le nombre de bits par élement, ce qui permet un élement max de 2^(2^3) - 1 = 255 dans la
            # matrice de DCT quantifiée
            # 1 bit pour le signe et le nombre de bits nécessaire pour encoder la valeur maximale du bloc en valeur
            # absolue
            non_zero_val_bits.append(3 + len(non_zero_val_list) * (1 + bits(max(np.abs(np.array(non_zero_val_list))))))

    non_zero_pos_bits = []
    for non_zero_pos_list in non_zero_pos_matrix.flatten():
        non_zero_pos_bits.append(len(non_zero_pos_list) * pos_encoding(np.abs(np.array(non_zero_pos_list))))

    N = sum(non_zero_pos_bits) + sum(non_zero_val_bits)
    return N


def image_reconstruction(DCT_q):
    """ Reconstruit une image à partir d'une DCT quantifiée """
    I_rec = np.zeros(DCT_q.shape)
    H, W = I_rec.shape

    for i in range(0, H, 8):
        for j in range(0, W, 8):
            I_rec[i:i + 8, j:j+8] = cv2.idct(DCT_q[i:i + 8, j:j+8] * Z)
    I_rec = np.clip(I_rec, 0, 255).astype(np.uint8)
    return I_rec


def DCT_reconstruction(non_zero_val_matrix, non_zero_pos_matrix):
    """ Reconstruit une matrice de l'ensemble des DCT quantifiée 8x8 à partir de :
    non_zero_val_matrix : liste de liste comportant les coefficients non nuls de chaque bloc
    non_zero_pos_matrix : liste de liste comportant les positions des coefficients non nuls de chaque bloc, encodée
    en plage
    """
    H, W = non_zero_val_matrix.shape  # val et L sont de taille (H/8, W/8) avec H,W la taille de l'image de base
    H *= 8
    W *= 8
    DCT = np.zeros((H, W))
    zz_coords = zigzag(8, 8)

    last_top_left_val = 0  # Valeur du dernier coin sup. gauche (pour codage prédictif)

    for i in range(int(H/8)):
        for j in range(int(W/8)):
            block_non_zero_pos = un_plage(non_zero_pos_matrix[i, j])
            block_non_zero_values = non_zero_val_matrix[i, j]

            assert len(np.where(np.array(block_non_zero_pos) == 1)[0]) == len(block_non_zero_values)
            n = 0  # Nombre de valeurs non-nulles traitées
            top_left = True  # Booléen indiquant si la valeur en train d'être traitée est un coin sup. gauche de bloc
            for k, coord in enumerate(zz_coords):
                if encode_diff:
                    # Avec codage prédictif
                    value = 0  # Valeur du pixel
                    if top_left:
                        # Les coins sup. gauche sont codés comme la différence de leur valeur à celle du coin sup.
                        # gauche précédent
                        value = last_top_left_val

                    if block_non_zero_pos[k]:
                        # Si la valeur à cette position est non-nulle :
                        value += block_non_zero_values[n]  # mise à jour de la valeur
                        n += 1  # Incrément du compteur de valeurs non-nulles traitées

                    if top_left:
                        # Mise à jour de la dernière valeur du coin sup. gauche
                        last_top_left_val = value

                    # Mise à jour de la matrice
                    DCT[coord[1] + 8*i, coord[0] + 8*j] = value
                    top_left = False
                else:
                    # Sans codage prédictif
                    if block_non_zero_pos[k]:
                        DCT[coord[1] + 8 * i, coord[0] + 8 * j] = block_non_zero_values[n]
                        n += 1
    return DCT


def compression(I):
    """ Retourne la version compressée de I et le taux de compression associé """
    global Z
    H, W = I.shape
    If = I.astype(np.float32)

    non_zero_val_matrix = []  # Matrice H/8 x W/8 contenant les listes de valeurs quantifiées non nulles
    non_zero_pos_matrix = []  # Matrice H/8 x W/8 contenant les listes de 0 et 1 indiquant les positions des valeurs non nulles
    zz_coords = zigzag(8, 8)
    last_top_left_val = 0  # Valeur du coin sup. gauche précédent (pour codage prédictif)

    for i in range(0, H, 8):
        line_non_zero_val = []
        line_non_zero_pos = []
        for j in range(0, W, 8):
            dct = cv2.dct(If[i:i+8, j:j+8])  # DCT d'un bloc 8x8 de l'image
            T = np.fix(dct/Z)  # Quantification

            block_non_zero_pos = []
            block_non_zero_val = []
            for coord in zz_coords:
                # Parcours en zigzag
                x = coord[0]
                y = coord[1]
                value = T[y, x]

                if encode_diff:
                    # Si encodage prédictif : calcul de la différence au coin sup. gauche précédent pour les coins
                    # sup. gauches
                    if x == y == 0:
                        value -= last_top_left_val
                        last_top_left_val = T[y, x]

                non_zero = int(value != 0)  # 0 si valeur nulle, 0 sinon
                block_non_zero_pos.append(non_zero)

                if non_zero:
                    # Récupération de la valeur si elle n'est pas nulle
                    block_non_zero_val.append(value)

            line_non_zero_pos.append(plage(block_non_zero_pos))  # Codage en plage des positions
            line_non_zero_val.append(block_non_zero_val)

        non_zero_pos_matrix.append(line_non_zero_pos)
        non_zero_val_matrix.append(line_non_zero_val)

    non_zero_pos_matrix = np.array(non_zero_pos_matrix, dtype=object)
    non_zero_val_matrix = np.array(non_zero_val_matrix, dtype=object)

    # Calcul du taux de compression
    size = calc_size(non_zero_val_matrix.flatten(), non_zero_pos_matrix.flatten())
    taux_comp = H * W * 8 / size

    # Reconstruction de la matrice de blocs de DCT
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
    plt.title('Histogramme de distribution des intensités')

    I_rec, taux_comp = compression(I)
    diff = np.abs(I_rec.astype(np.int16) - I.astype(np.int16)).astype(np.uint8)
    cv2.imwrite("res_dct.png", I_rec)
    cv2.imwrite("diff_dct.png", diff)

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
    plt.imshow(diff, "gray")
    plt.title('Différence')
    plt.show()

    cv2.imwrite("oreille.png", I[10:190, 170:300])
    cv2.imwrite("oreille_dct.png", I_rec[10:190, 170:300])


def test_rand():
    """ Fonction qui génère des variations aléatoires de Z pour chercher une version plus efficace """
    global Z
    I = cv2.imread("Images/04.png", 0)
    I_rec, taux_comp = compression(I)
    best = taux_comp

    Z_i = Z
    best_Z = Z_i
    base_RMSE = RMSE(I, I_rec)

    n = 5
    for i in range(100):
        r = np.fix(np.random.rand(8, 8) * 2 * (n + 1) - (n + 1))
        Z = Z_i + r
        Z[np.where(Z <= 0)] = 1
        I_rec, taux_comp = compression(I)
        err = RMSE(I, I_rec)
        if taux_comp > best and err < base_RMSE:
            best = taux_comp
            best_Z = Z
            print(best, err)
        print(i/100)
    print(best)
    print(best_Z)


if __name__ == "__main__":
    """ z_fac pour atteindre 15 suivant le mode utilisé :
    Pas d'encodage prédictif et Z de base : 3.06
    Encodage prédictif et Z de base : 2.14
    Pas d'encodage prédictif et Z2 : 2.79
    Encodage prédictif et Z2 : 1.91 """
    use_Z2 = False  # Utilisation de la matrice modifiée Z2 plutôt que de la version JPEG Z
    encode_diff = True  # Utilisation de l'encodage prédictif
    z_fac = 2.14  # Facteur de multiplication de la matrice Z
    if use_Z2:
        Z = Z2
    Z = np.round(Z.astype(np.float32) * z_fac).astype(np.uint16)
    main()
