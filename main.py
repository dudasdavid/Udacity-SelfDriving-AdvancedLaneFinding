import pickle
import cv2
import helper_functions as utils
from argparse import ArgumentParser
import time
import numpy as np


def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = ArgumentParser()

    parser.add_argument("-i", "--input", required=False, type=str, default="project_video.mp4",
                        help="Path to image or video file")

    return parser

# read camera calibration parameters
dist_pickle = pickle.load(open("camera_cal/cal_pickle.p", "rb"))
mtx = dist_pickle["mtx"]
dist = dist_pickle["dist"]

#Set hist parameters
hist_height = 64
hist_width = 256
nbins = 32
bin_width = hist_width/nbins




def process_frame(img, left_lane, right_lane):
    # get frame dimensions for later use
    w, h = img.shape[1], img.shape[0]

    # first, let's undistort the image
    undistorted = utils.undistort_image(img, mtx, dist)

    # define roi and destination region that will be used for bird's perspective
    src_region = np.array([[[20, h], [550, 440], [w - 550, 440], [w - 20, h]]]) ### first project video
    src_region = np.array([[[140, h], [550, 475], [w - 550, 475], [w - 140, h]]]) ### challenge video
    #src_region = np.array([[[20, h], [300, 480], [w - 300, 480], [w - 20, h]]])

    dst_region = np.array([[[360, h], [0, 0], [w - 0, 0], [w - 360, h]]])

    # show selected regions on a copy image
    region_img = np.copy(undistorted)
    cv2.polylines(region_img, src_region, True, (0, 0, 255), 5)
    cv2.polylines(region_img, dst_region, True, (0, 255, 0), 2)

    # calculate the transformation matrices for bird's perspective transformations
    M, M_inv = utils.calc_transformation_matrices(src_region, dst_region)

    # create a mask according to roi
    gray = utils.convert_to_gray(img)
    mask = np.zeros((h, w), dtype=np.uint8)
    ignore_mask_color = 255
    cv2.fillPoly(mask, src_region, ignore_mask_color)
    # and create a 3 channel mask, too, if needed
    color_mask = np.dstack((mask, mask, mask))

    # calculate 1 channel images in HLS representation
    H, L, S = utils.convert_to_hls(undistorted)
    # apply binary threshold on S channel
    #S = utils.threshold_binary(S, (50, 220)) ### project video parameter
    #S = utils.threshold_binary(S, (20, 150)) ### challenge video
    S = utils.threshold_binary(S, (20, 255))

    # apply roi mask on S channel
    masked_S = cv2.bitwise_and(S, mask)

    # apply various gradient thresholds
    # choose a Sobel kernel size first
    ksize = 17  # Choose a larger odd number to smooth gradient measurements

    # Apply each of the thresholding functions
    #gradx = utils.abs_sobel_thresh(undistorted, orient='x', sobel_kernel=ksize, thresh=(50, 150))
    gradx = utils.abs_sobel_thresh(undistorted, orient='x', sobel_kernel=ksize, thresh=(60, 110))
    grady = utils.abs_sobel_thresh(undistorted, orient='y', sobel_kernel=ksize, thresh=(20, 110))
    #mag_binary = utils.mag_thresh(undistorted, sobel_kernel=ksize, mag_thresh=(100, 200)) ### project video parameter
    mag_binary = utils.mag_thresh(undistorted, sobel_kernel=ksize, mag_thresh=(50, 200))
    dir_binary = utils.dir_threshold(undistorted, sobel_kernel=ksize, thresh=(np.pi / 4, np.pi / 2))

    # combine all gradient thresholds
    combined = np.zeros((h, w), dtype=np.uint8)
    combined[(gradx == 1) | ((grady == 1) & (mag_binary == 1) & (dir_binary == 1))] = 1

    # apply roi mask on gradient thresholds
    masked_combined = cv2.bitwise_and(combined, mask)

    # stack gradient thresholds and color filter
    result, result_binary = utils.stack_binaries(masked_combined, masked_S)

    # convert result to bird's view perspective
    birdseye = utils.warp_transform(result_binary*255, M)

    # calculate histogram from the bottom half of the bird's view
    histogram = utils.hist(birdseye/255)

    # Create an empty image for the histogram
    hist = utils.draw_histogram(histogram, 64)

    if left_lane.reliable and right_lane.reliable:

        leftx, lefty = utils.search_around_poly(birdseye, left_lane.get_poly())
        rightx, righty = utils.search_around_poly(birdseye, right_lane.get_poly())

        left_lane.update_coordintes(leftx, lefty)
        right_lane.update_coordintes(rightx, righty)

        out_img = birdseye.copy()

    else:
        leftx, lefty, rightx, righty, out_img = utils.find_lane_pixels(birdseye)

        left_lane.update_coordintes(leftx, lefty)
        right_lane.update_coordintes(rightx, righty)

    start_width = right_lane.fitx[0] - left_lane.fitx[0]
    end_width = right_lane.fitx[-1] - left_lane.fitx[-1]
    left_curvature, right_curvature, horizontal_offset = utils.calculate_lane_curvature(left_lane, right_lane)

    if (left_curvature > 0 and right_curvature < 0) or (left_curvature < 0 and right_curvature > 0):
        left_lane.recovery()
        right_lane.recovery()

    if abs(start_width - end_width) > 100 or abs(start_width) < 600 or abs(end_width) < 400:
        left_lane.recovery()
        right_lane.recovery()

    out_img = utils.draw_lane_pixels(out_img, left_lane, color=(0, 0, 255))
    out_img = utils.draw_lane_pixels(out_img, right_lane)
    out_img = utils.draw_poly(out_img, left_lane, width=5)
    out_img = utils.draw_poly(out_img, right_lane, width=5)

    result = utils.draw_lane_area(birdseye, undistorted, left_lane, right_lane, M_inv)
    result = utils.draw_lane_lines(birdseye, result, left_lane, right_lane, M_inv, 10)

    # stack small images to the original frame
    result = utils.add_small_pictures(result, [region_img, birdseye, hist, out_img, masked_combined*255])
    return result



def main():
    # Grab command line args
    args = build_argparser().parse_args()

    # Handle the input stream
    image_flag = False
    # Check if the input is a webcam
    if args.input == 'CAM':
        args.input = 0
    elif args.input.endswith('.jpg') or args.input.endswith('.bmp') or args.input.endswith('.png'):
        image_flag = True
    elif args.input.endswith('.mp4'):
        # video file is accepted
        pass
    else:
        print("ERROR: Invalid input, it must be CAM, image (.jpg, .bmp or .png) or video (.mp4)!")
        raise NotImplementedError

    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    # get video FPS data
    fps = cap.get(cv2.CAP_PROP_FPS)

    width = int(cap.get(3))
    height = int(cap.get(4))

    # Create a video writer for the output video
    if not image_flag:
        # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
        # on Mac, and `0x00000021` on Linux
        out = cv2.VideoWriter('out.mp4', 0x7634706d, fps, (width, height))
    else:
        out = None

    fps_marking = True

    left_lane = utils.Lane()
    right_lane = utils.Lane()
    # Loop until stream is over
    while cap.isOpened():

        # Read from the video capture
        ret, frame = cap.read()

        if not ret:
            break

        # start measuring overall execution time
        start_processing_time = time.time()

        output = process_frame(frame, left_lane, right_lane)

        # Measure overall FPS
        total_processing_time = time.time() - start_processing_time
        total_fps = 1 / (total_processing_time)

        # if FPS marking run time switch is turned on print some details on the image
        if fps_marking:
            label_text = f"FPS: {total_fps:.3}"
            cv2.putText(output, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        #print(utils.calculate_lane_curvature(left_lane, right_lane))
        if len(left_lane.x) > 0:
            left_curvature, right_curvature, horizontal_offset = utils.calculate_lane_curvature(left_lane, right_lane)
            label_text = f"Left lane curvature: {np.absolute(left_curvature):.1f} m"
            cv2.putText(output, label_text, (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            label_text = f"Right lane curvature: {np.absolute(right_curvature):.1f} m"
            cv2.putText(output, label_text, (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            label_text = f"Lane center offset: {horizontal_offset:.1f} m"
            cv2.putText(output, label_text, (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # Show the output image and save the output video
        cv2.imshow('Frame', output)
        if not image_flag:
            out.write(output)
        else:
            # Write an output image if `single_image_mode`
            cv2.imwrite('output_image.jpg', frame)

        ret = cv2.waitKey(1)
        if ret & 0xFF == ord('q'):
            break

    if not image_flag:
        out.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()


