import cv2
import numpy as np
from matplotlib import pyplot as plt
from couprie.skelpar import lhthinpar, lhthinpar_asymmetric
from couprie.jitversion.llambdakern import llambdakern

def main():
    img_path = "../test_images/test_2.png"
    original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if original is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {img_path}")

    # Копия для обработки
    image = original.copy()

    image = lhthinpar(image)
    image = lhthinpar_asymmetric(image)
    image = llambdakern(image, 20)

    # Отображение
    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(original, cmap="gray")
    ax1.set_title("Исходное изображение")
    ax1.axis("off")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(image, cmap="gray")
    ax2.set_title("Конечное изображение")
    ax2.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()