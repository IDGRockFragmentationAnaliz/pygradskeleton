# py-grayscale-skeletonization


```python
import cv2
from pygradskeleton import grayscaleskelet

image = cv2.imread("path/to/image")
result_1 = grayscaleskelet(image, method="KIM")
```

```commandline
pip intsall numpy
pip install opencv-contrib-python
pip install numba
```

## Optional
```commandline
pip install matplotlib
```