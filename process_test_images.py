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
        if (not image_.endswith('.jpg')) or "undistorted" in image_ or "figure" in image_:
            continue
        # Read in and grayscale the image
        image = mpimg.imread('test_images/' + image_)
        w, h = image.shape[1], image.shape[0]

        undistorted = utils.undistort_image(image, mtx, dist)

        mpimg.imsave('test_images/undistorted_' + image_, undistorted)


        src_region = np.array([[[20, h], [550, 440], [w - 550, 440], [w - 20, h]]])
        dst_region = np.array([[[360, h], [0, 0], [w-0, 0], [w-360, h]]])

        M, M_inv = utils.calc_transformation_matrices(src_region, dst_region)

        # Show: Selected Region
        region_img = np.copy(undistorted)
        cv2.polylines(region_img, src_region, True, (0, 0, 255), 5)
        cv2.polylines(region_img, dst_region, True, (0, 255, 0), 2)

        birdseye = utils.warp_transform(undistorted, M)
        cv2.polylines(birdseye, np.array([[[370, h], [370, 0], [w-370, 0], [w-370, h]]]), True, (0, 255, 0), 2)

        # Plot the result
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 9))
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
        plt.savefig('test_images/camera_figures_' + image_)
        plt.close()

        break

        gray = utils.convert_to_gray(undistorted)
        R, G, B = utils.convert_to_rgb(undistorted)
        hls_H, hls_L, hls_S = utils.convert_to_hls(undistorted)
        hsv_H, hsv_S, hsv_V = utils.convert_to_hsv(undistorted)


        # Plot the result
        f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(R, cmap='gray')
        ax1.set_title('R', fontsize=fs)
        ax2.imshow(G, cmap='gray')
        ax2.set_title('G', fontsize=fs)
        ax3.imshow(B, cmap='gray')
        ax3.set_title('B', fontsize=fs)

        ax4.imshow(hls_H, cmap='gray')
        ax4.set_title('HLS - H', fontsize=fs)
        ax5.imshow(hls_L, cmap='gray')
        ax5.set_title('HLS - L', fontsize=fs)
        ax6.imshow(hls_S, cmap='gray')
        ax6.set_title('HLS - S', fontsize=fs)

        ax7.imshow(hsv_H, cmap='gray')
        ax7.set_title('HSV - H', fontsize=fs)
        ax8.imshow(hsv_S, cmap='gray')
        ax8.set_title('HSV - S', fontsize=fs)
        ax9.imshow(hsv_V, cmap='gray')
        ax9.set_title('HSV - V', fontsize=fs)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        #plt.show()
        plt.savefig('test_images/color_figures_' + image_)
        plt.close()

        ### HLS's S channel is the best bet for both white and yellow lines
        S_th = utils.threshold_binary(hls_S, (90, 255))

        # Display the image
        # Plot the result
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(hls_S, cmap='gray')
        ax1.set_title('HLS - S', fontsize=fs)
        ax2.imshow(S_th, cmap='gray')
        ax2.set_title('Thresholded S', fontsize=fs)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        #plt.show()
        plt.savefig('test_images/color_th_figures_' + image_)
        plt.close()

        # Choose a Sobel kernel size
        ksize = 17  # Choose a larger odd number to smooth gradient measurements

        # Run the functions
        # Apply each of the thresholding functions
        gradx = utils.abs_sobel_thresh(undistorted, orient='x', sobel_kernel=ksize, thresh=(20, 110))
        grady = utils.abs_sobel_thresh(undistorted, orient='y', sobel_kernel=ksize, thresh=(20, 110))
        mag_binary = utils.mag_thresh(undistorted, sobel_kernel=ksize, mag_thresh=(100, 200))
        dir_binary = utils.dir_threshold(undistorted, sobel_kernel=ksize, thresh=(np.pi/4, np.pi/2))

        combined = np.zeros_like(dir_binary)
        combined[(gradx == 1) | ((grady == 1) & (mag_binary == 1) & (dir_binary == 1))] = 1
        # Plot the result
        f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(undistorted)
        ax1.set_title('Original Image', fontsize=fs)
        ax2.imshow(combined, cmap='gray')
        ax2.set_title('Combined', fontsize=fs)

        ax3.imshow(gradx, cmap='gray')
        ax3.set_title('Thresholded Gradient X', fontsize=fs)
        ax4.imshow(grady, cmap='gray')
        ax4.set_title('Thresholded Gradient Y', fontsize=fs)
        ax5.imshow(mag_binary, cmap='gray')
        ax5.set_title('Thresholded Magnitude', fontsize=fs)
        ax6.imshow(dir_binary, cmap='gray')
        ax6.set_title('Thresholded Grad. Dir.', fontsize=fs)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        #plt.show()
        plt.savefig('test_images/gradient_th_figures_' + image_)
        plt.close()

        result, result_binary = utils.stack_binaries(combined, S_th)

        # Plot the result
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
        f.tight_layout()

        ax1.imshow(undistorted)
        ax1.set_title('Original Image', fontsize=fs)

        ax2.imshow(result)
        ax2.set_title('Pipeline Result', fontsize=fs)

        ax3.imshow(result_binary, cmap='gray')
        ax3.set_title('Pipeline Result', fontsize=fs)

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        # plt.show()
        plt.savefig('test_images/pipeline_figures_' + image_)
        plt.close()

        birdseye_binary = utils.warp_transform(result_binary, M)
        histogram = utils.hist(birdseye_binary/255)

        # Plot the result
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
        f.tight_layout()

        ax1.imshow(birdseye)
        ax1.set_title('Original Image', fontsize=fs)

        ax2.imshow(birdseye_binary, cmap='gray')
        ax2.set_title('Pipeline Result', fontsize=fs)

        ax3.plot(histogram)
        ax3.set_title("Histogram Of Pixel Intensities (Image Bottom Half)")

        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()
        plt.savefig('test_images/birdseye_figures_' + image_)
        plt.close()

        #break

def main():
    process_test_images()


if __name__ == '__main__':
    main()












