import cv2
import numpy as np


def undistort_image(img, mtx, dist):
    undist = cv2.undistort(img, mtx, dist, None, mtx)

    return undist

def calc_transformation_matrices(src, dst):
    M     = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))
    M_inv = cv2.getPerspectiveTransform(np.float32(dst), np.float32(src))

    return M, M_inv

def warp_transform(img, M):
    w, h = img.shape[1], img.shape[0]
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR)
