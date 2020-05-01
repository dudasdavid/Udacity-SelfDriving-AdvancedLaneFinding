import pickle
import cv2
import os
import helper_functions as utils
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

dist_pickle = pickle.load(open("camera_cal/cal_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

fs = 20

def process_test_images():
    image_list = os.listdir("test_images/")

    for image_ in image_list:
        if (not image_.endswith('.jpg')) or "undistorted" in image_:
            continue
        # Read in and grayscale the image
        image = mpimg.imread('test_images/' + image_)
        w, h = image.shape[1], image.shape[0]

        undistorted = utils.undistort_image(image, mtx, dist)

        mpimg.imsave('test_images/undistorted_' + image_, undistorted)


        src_region = np.array([[[-400, h], [530, 450], [w-530, 450], [w+400, h]]])
        dst_region = np.array([[[0, h], [0, 0], [w-0, 0], [w-0, h]]])

        M, M_inv = utils.calc_transformation_matrices(src_region, dst_region)

        # Show: Selected Region
        region_img = np.copy(undistorted)
        cv2.polylines(region_img, src_region, True, (0, 0, 255), 5)
        cv2.polylines(region_img, dst_region, True, (0, 255, 0), 2)

        birdseye = utils.warp_transform(undistorted, M)
        cv2.polylines(birdseye, np.array([[[370, h], [370, 0], [w-370, 0], [w-370, h]]]), True, (0, 255, 0), 2)


        # Plot the result
        f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=fs)
        ax2.imshow(undistorted, cmap='gray')
        ax2.set_title('Undistorted', fontsize=fs)

        ax3.imshow(region_img, cmap='gray')
        ax3.set_title('Region', fontsize=fs)
        ax4.imshow(birdseye, cmap='gray')
        ax4.set_title('Birdseye', fontsize=fs)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

        #break

def main():
    process_test_images()


if __name__ == '__main__':
    main()












