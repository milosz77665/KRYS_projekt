import cv2
import matplotlib.pyplot as plt


def calculate_histogram(gray_image):
    histogram = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
    histogram = histogram.flatten()
    return histogram


def plot_histograms(hist_normal, hist_stego):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.plot(hist_normal, color='blue')
    plt.title("Normalny obraz")
    plt.xlabel("Wartość pikseli")
    plt.ylabel("Częstotliwość")
    plt.grid(True)

    plt.subplot(1, 3, 2)
    plt.plot(hist_stego, color='green')
    plt.title("Obraz z ukrytą wiadomością")
    plt.xlabel("Wartość pikseli")
    plt.ylabel("Częstotliwość")
    plt.grid(True)

    diff_hist = hist_normal - hist_stego
    plt.subplot(1, 3, 3)
    plt.plot(diff_hist, color='red')
    plt.title("Różnica histogramów")
    plt.xlabel("Wartość pikseli")
    plt.ylabel("Różnica częstotliwości")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def histogram_analysis(cover_image_path, stego_image_path):
    cover_image = cv2.imread(cover_image_path, cv2.IMREAD_GRAYSCALE)
    stego_image = cv2.imread(stego_image_path, cv2.IMREAD_GRAYSCALE)

    hist_cover = calculate_histogram(cover_image)
    hist_stego = calculate_histogram(stego_image)

    plot_histograms(hist_cover, hist_stego)


if __name__ == "__main__":
    # cover_image_path = './cover_images/1.jpg'
    # stego_image_path = './LSB/LSB_1.jpg'
    # histogram_analysis(cover_image_path, stego_image_path)
    for i in range(1, 11):
        cover_image_path = f'./cover_images/{i}.png'
        stego_image_path = f'./LSB/LSB_{i}.png'
        print(cover_image_path)
        print(stego_image_path)

        histogram_analysis(cover_image_path, stego_image_path)
