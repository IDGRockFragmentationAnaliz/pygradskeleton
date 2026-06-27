import cv2
import time
import numpy as np
from matplotlib import pyplot as plt
from couprie.skelpar import lhthinpar, lhthinpar_asymmetric
from couprie.llambdakern import llambdakern
from couprie.jitversion.thin_segment import thin_segment
from couprie.jitversion.crestrestoration import crestrestore

def main():
    img_path = "../test_images/test_1.png"
    original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if original is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {img_path}")

    # Копия для обработки
    # image = original.copy()
    # t = time.perf_counter()
    # image_thin = lhthinpar(image, progress=True)
    # print("lhthinpar:", time.perf_counter() - t)
    #
    # t = time.perf_counter()
    # image_thin = lhthinpar_asymmetric(image_thin, progress=True)
    # print("lhthinpar_asymmetric", time.perf_counter() - t)
    #
    # # Сохраняем промежуточный результат перед llambdakern
    cache_path = "../test_images/image_before_llambdakern.png"
    # cv2.imwrite(cache_path, image_thin)

    # Тут же загружаем его обратно
    image_thin = cv2.imread(cache_path, cv2.IMREAD_GRAYSCALE)

    if image_thin is None:
        raise FileNotFoundError(f"Не удалось загрузить промежуточное изображение: {cache_path}")

    t = time.perf_counter()
    image_lamb = llambdakern(image_thin, 20, progress=True)
    print("llambdakern", time.perf_counter() - t)
    #
    t = time.perf_counter()
    image_skel = thin_segment(image_lamb)
    print("thin_segment", time.perf_counter() - t)

    # Отображение
    fig = plt.figure(figsize=(7, 7))

    ax0 = fig.add_subplot(2, 2, 1)
    ax0.imshow(original, cmap="gray")
    ax0.set_title("Исходное изображение")
    ax0.axis("off")

    ax1 = fig.add_subplot(2, 2, 2)
    ax1.imshow(image_thin, cmap="gray")
    ax1.set_title("Процесс утоньшения границ")
    ax1.axis("off")
    ax1.sharex(ax0)
    ax1.sharey(ax0)

    ax2 = fig.add_subplot(2, 2, 4)
    ax2.imshow(image_lamb, cmap="gray")
    ax2.set_title("Лямбда-левеленг")
    ax2.axis("off")
    ax2.sharex(ax0)
    ax2.sharey(ax0)
    #
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(image_skel, cmap="gray")
    ax3.set_title("Конечное изображение")
    ax3.axis("off")
    ax3.sharex(ax0)
    ax3.sharey(ax0)

    #ax0.set_xlim([230, 400])
    #ax0.set_ylim([25, 195])

    plt.tight_layout()
    #plt.savefig("../pictures/example.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()