import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_image(path: str):
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Görüntü bulunamadı: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return img_rgb, gray

def clip_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img, 0, 255).astype(np.uint8)

# --- Uzamsal Yumuşatma (gürültü azaltma) ---
def gaussian_blur(gray: np.ndarray, ksize=5, sigma=1.2):
    if ksize % 2 == 0:
        ksize += 1
    return cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=sigma)

def median_blur(gray: np.ndarray, ksize=5):
    if ksize % 2 == 0:
        ksize += 1
    return cv2.medianBlur(gray, ksize)

def bilateral_filter(gray: np.ndarray, d=9, sigmaColor=60, sigmaSpace=60):
    return cv2.bilateralFilter(gray, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)

# --- Keskinleştirme 1: Laplacian (2. türev) ---
def sharpen_laplacian(gray: np.ndarray, ksize=3, alpha=1.0):
    # Laplacian kenarları çıkarır; orijinale ekleyip keskinleştirme yaparız
    lap = cv2.Laplacian(gray, ddepth=cv2.CV_32F, ksize=ksize)
    sharp = gray.astype(np.float32) - alpha * lap
    return clip_uint8(sharp)

# --- Keskinleştirme 2: Unsharp Mask (en yaygın ve temiz yöntem) ---
def unsharp_mask(gray: np.ndarray, ksize=5, sigma=1.2, amount=1.5, threshold=0):
    """
    amount: 1.0-2.5 arası genelde iyi
    threshold: düşük kontrast bölgeleri keskinleştirmeyi azaltır (0 = kapalı)
    """
    blurred = gaussian_blur(gray, ksize=ksize, sigma=sigma).astype(np.float32)
    gray_f = gray.astype(np.float32)
    mask = gray_f - blurred  # detay/kenar bilgisi
    sharp = gray_f + amount * mask

    if threshold > 0:
        low_contrast = np.abs(mask) < threshold
        sharp[low_contrast] = gray_f[low_contrast]

    return clip_uint8(sharp)

# --- Keskinleştirme 3: High-Boost Filtering (Unsharp'ın geneli) ---
def high_boost(gray: np.ndarray, ksize=5, sigma=1.2, A=1.8):
    """
    A > 1 ise high-boost. (A=1 => unsharp mask benzeri)
    """
    blurred = gaussian_blur(gray, ksize=ksize, sigma=sigma).astype(np.float32)
    gray_f = gray.astype(np.float32)
    mask = gray_f - blurred
    boosted = gray_f + (A - 1.0) * mask
    return clip_uint8(boosted)

# --- Keskinleştirme 4: Sobel tabanlı kenar + ekleme (1. türev) ---
def sharpen_sobel(gray: np.ndarray, alpha=0.6):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)  # kenar şiddeti
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    sharp = gray.astype(np.float32) + alpha * mag
    return clip_uint8(sharp), clip_uint8(mag)

def show_results(gray: np.ndarray, results: dict, cols=3, title="Uzamsal Filtreleme Sonuçları"):
    n = len(results) + 1
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(14, 8))
    plt.suptitle(title)
    # Orijinal
    plt.subplot(rows, cols, 1)
    plt.imshow(gray, cmap="gray")
    plt.title("Orijinal (Gray)")
    plt.axis("off")

    i = 2
    for name, img in results.items():
        plt.subplot(rows, cols, i)
        plt.imshow(img, cmap="gray")
        plt.title(name)
        plt.axis("off")
        i += 1

    plt.tight_layout()
    plt.show()

def main():
    image_path = "input.jpg"  # <-- fotoğrafın dosya adı / yolu
    _, gray = read_image(image_path)

    # Önce gürültü azaltma örnekleri:
    g_blur = gaussian_blur(gray, ksize=5, sigma=1.2)
    m_blur = median_blur(gray, ksize=5)
    b_flt = bilateral_filter(gray, d=9, sigmaColor=60, sigmaSpace=60)

    # Keskinleştirme örnekleri:
    lap_sharp = sharpen_laplacian(gray, ksize=3, alpha=1.0)
    unsharp = unsharp_mask(gray, ksize=5, sigma=1.2, amount=1.7, threshold=3)
    hboost = high_boost(gray, ksize=5, sigma=1.2, A=2.0)
    sobel_sharp, sobel_edges = sharpen_sobel(gray, alpha=0.6)

    results = {
        "Gaussian Blur (5, σ=1.2)": g_blur,
        "Median Blur (5)": m_blur,
        "Bilateral (d=9)": b_flt,
        "Laplacian Sharp": lap_sharp,
        "Unsharp Mask": unsharp,
        "High-Boost (A=2.0)": hboost,
        "Sobel Edges": sobel_edges,
        "Sobel Sharp": sobel_sharp,
    }

    show_results(gray, results, cols=3)

    # İstersen kaydet:
    cv2.imwrite("out_unsharp.png", unsharp)
    cv2.imwrite("out_laplacian.png", lap_sharp)
    cv2.imwrite("out_highboost.png", hboost)
    cv2.imwrite("out_sobel_sharp.png", sobel_sharp)
    print("Kaydedildi: out_*.png")

if __name__ == "__main__":
    main()
