import os
import cv2
import numpy as np


def discrimination_function(img_window):
    img_window = np.concatenate([
        np.diagonal(img_window[::-1, :], k)[::(2 * (k % 2) - 1)]
        for k in range(1 - img_window.shape[0], img_window.shape[0])
    ])
    return np.sum(np.abs(img_window[:-1] - img_window[1:]))


def support_f_1(img_window):
    img_window = np.copy(img_window)
    even_values = img_window % 2 == 0
    img_window[even_values] += 1
    img_window[~even_values] -= 1
    return img_window


def flipping_operation(img_window, mask):
    def f_1(x): return support_f_1(x)
    def f_0(x): return np.copy(x)
    def f_neg1(x): return support_f_1(x + 1) - 1

    result = np.empty(img_window.shape)
    flip_functions = {-1: f_neg1, 0: f_0, 1: f_1}

    for key in flip_functions:
        indices = np.where(mask == key)
        result[indices] = flip_functions[key](img_window[indices])

    return result


def calculate_count_groups(img, mask):
    count_reg, count_sing, count_unusable = 0, 0, 0

    for ih in range(0, img.shape[0], mask.shape[0]):
        for iw in range(0, img.shape[1], mask.shape[1]):
            img_window = img[ih: ih + mask.shape[0], iw: iw + mask.shape[1]]
            flipped_output = flipping_operation(img_window, mask)

            discrimination_original = discrimination_function(img_window)
            discrimination_flipped = discrimination_function(flipped_output)

            if discrimination_flipped > discrimination_original:
                count_reg += 1
            elif discrimination_flipped < discrimination_original:
                count_sing += 1
            else:
                count_unusable += 1

    total_groups = count_reg + count_sing + count_unusable
    if total_groups == 0:
        return 0, 0

    return count_reg / total_groups, count_sing / total_groups


def rs_analysis(stego_image_path):
    mask = np.random.randint(low=0, high=2, size=(8, 8))

    stego_image = cv2.cvtColor(cv2.imread(stego_image_path),
                               cv2.COLOR_BGR2RGB).astype('int16')

    img_w = stego_image.shape[0]
    img_h = stego_image.shape[1]

    if img_w % 8 != 0:
        img_w = img_w + (8 - img_w % 8)
    if img_h % 8 != 0:
        img_h = img_h + (8 - img_h % 8)

    stego_image = cv2.resize(stego_image, (img_h, img_w),
                             interpolation=cv2.INTER_AREA)

    rm, sm = calculate_count_groups(stego_image[:, :, 0], mask)
    r_neg_m, s_neg_m = calculate_count_groups(stego_image[:, :, 0], - mask)

    print('\nAnaliza RS:')
    print(
        f'Rm: {rm}, Sm: {sm}, R-m: {r_neg_m}, S-m: {s_neg_m}')

    diff = (r_neg_m - s_neg_m) - (rm - sm)

    if diff >= 0.1:
        print("\nW tym obrazie prawdopodobnie została ukryta wiadomość, ponieważ R-m - S-m > Rm - Sm")
        print(
            f"Różnica (R-m - S-m) - (Rm - Sm) wynosi {(r_neg_m - s_neg_m) - (rm - sm)}")
    elif diff < -0.1:
        print("\nW tym obrazie prawdopodobnie nie została ukryta wiadomość, ponieważ R-m - S-m <= Rm - Sm")
        print(
            f"Różnica (R-m - S-m) - (Rm - Sm) wynosi {(r_neg_m - s_neg_m) - (rm - sm)}")
    elif diff > 0:
        print("\nAnaliza wskazuje, że w obrazie została ukryta wiadomość, ponieważ  R-m - S-m > Rm - Sm, jednak różnica jest bardzo mała")
        print(
            f"Różnica (R-m - S-m) - (Rm - Sm) wynosi {(r_neg_m - s_neg_m) - (rm - sm)}")
    else:
        print("\nAnaliza wskazuje, że w obrazie nie została ukryta wiadomość, ponieważ  R-m - S-m <= Rm - Sm, jednak różnica jest bardzo mała")
        print(
            f"Różnica (R-m - S-m) - (Rm - Sm) wynosi {(r_neg_m - s_neg_m) - (rm - sm)}")


if __name__ == "__main__":
    # stego_image_path ='./LSB/LSB_1.jpg'
    # rs_analysis(stego_image_path)
    directory = "./LSB"
    for entry in os.listdir(directory):
        stego_image_path = os.path.join(directory, entry)
        print(stego_image_path)

        rs_analysis(stego_image_path)
