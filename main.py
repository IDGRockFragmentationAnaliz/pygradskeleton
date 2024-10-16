from kim import Kim


methods = {
       "KIM": Kim
}


def grayscale_skeletonize(image, method="KIM"):
    return methods[method](image.copy()).run()
