import numpy as np
from kim import Kim


def main():
    img_path = "./test_images/test_2.png"
    image = cv2.imread("path/to/image")
    result_1 = grayscale_skeletonize(image, method="KIM")

if __name__ == "__main__":
    main()