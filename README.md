# py-grayscale-skeletonization


```python
import cv2
from py-grayscale-skeletonization import grayscaleskelet

image = cv2.imread("path/to/image")
result_1 = grayscaleskelet(image, method="KIM")
```