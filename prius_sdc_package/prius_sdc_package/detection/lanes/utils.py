import math
import numpy as np


def calculate_distance(pa, pb):
    return math.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)

def cord_sort(contours, order):
    if contours:
        contour = contours[0]
        # what is this reshape doing?
        contour = np.reshape(contour, (contour.shape[0], contour.shape[2]))
        
        order_list = []
        if order == "rows":
            order_list.append((0, 1))
        else:
            order_list.append((1, 0))
        index = np.lexsort((contour[:, order_list[0][0]], contour[:, order_list[0][1]]))
        sorted = contour[index]
        return sorted
    else:
        return contours