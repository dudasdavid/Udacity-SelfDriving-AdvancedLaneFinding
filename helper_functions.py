import cv2
import numpy as np


# Define a class to receive the characteristics of each line detection
class Lane:
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #x values for detected line pixels
        self.x = None
        #y values for detected line pixels
        self.y = None

        self.reliable = False

        self.ploty = None
        self.fitx = None
        self.imshape = (720, 1280, 3)

    def update_poly(self, fit):
        self.current_fit = fit
        self.best_fit = fit


    def get_poly(self):
        return self.current_fit

    def update_coordintes(self, x, y):
        self.x = x
        self.y = y

        fit = list(fit_poly(self.x, self.y))

        if self.reliable == True:
            fit[0] = self.filter_poly(fit[0], self.best_fit[0])
            fit[1] = self.filter_poly(fit[1], self.best_fit[1])
            fit[2] = self.filter_poly(fit[2], self.best_fit[2])

        self.ploty, self.fitx = calc_poly(self.imshape, fit)

        self.update_poly(fit)

        self.reliable = True


    def get_coordinates(self):
        return self.x, self.y

    def filter_poly(self, new, avg, a=0.1):
        avg = (a * new) + (1.0 - a) * avg
        return avg


def convert_to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def convert_to_rgb(img):
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    return R, G, B


def convert_to_hls(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    H = hls[:, :, 0]
    L = hls[:, :, 1]
    S = hls[:, :, 2]

    return H, L, S


def convert_to_hsv(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    H = hsv[:, :, 0]
    S = hsv[:, :, 1]
    V = hsv[:, :, 2]

    return H, S, V


def threshold_binary(img, thresh=(200, 255)):
    binary = np.zeros_like(img)
    binary[(img >= thresh[0]) & (img <= thresh[1])] = 1

    return binary


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


def abs_sobel_thresh(img, orient='y', sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output


# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output


def stack_binaries(gradient, color):
    # Stack each channel
    color_binary = np.dstack((gradient, color, np.zeros_like(gradient))) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(gradient)
    combined_binary[(color == 1) | (gradient == 1)] = 1

    return color_binary, combined_binary


def hist(img):
    # Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = img[img.shape[0] // 2:, :]

    # Sum across image pixels vertically - make sure to set an `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_half, axis=0)

    return histogram


def draw_histogram(data_array, nbins = 64):


    # Create an empty image for the histogram
    hist = np.zeros((240, 320), dtype=np.uint8)

    # Loop through each bin and plot the rectangle in white
    bin_width = int(320/nbins)

    last_processed = 0
    data_in_bins = int(len(data_array) / nbins)
    for x in range(nbins):
        average = sum(data_array[x*data_in_bins : x*data_in_bins+nbins])/nbins
        cv2.rectangle(hist, (int(x * bin_width), int(average)), (int(x * bin_width + bin_width - 1), 240), (255), -1)

    # Flip upside down
    hist = np.flipud(hist)

    return hist


def add_small_pictures(img, small_images, size=(220, 124)):
    """
    Returns a new image made up of the lane area image, and the remaining lane images are overlaid as
    small images in a row at the top of the the new image
    """
    x_base_offset = 20
    y_base_offset = 50

    x_offset = x_base_offset
    y_offset = y_base_offset

    for small in small_images:
        small = cv2.resize(small, size)
        if len(small.shape) == 2:
            small = np.dstack((small, small, small))

        img[y_offset: y_offset + size[1], x_offset: x_offset + size[0]] = small

        x_offset += size[0] + x_base_offset



    return img


def find_lane_pixels(binary_warped, nwindows = 9, margin = 100, minpix = 50):
    '''
    :param binary_warped:
    :param nwindows: the number of sliding windows
    :param margin: Set the width of the windows +/- margin
    :param minpix: Set minimum number of pixels found to recenter window
    :return:
    '''
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)

    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 10)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 10)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_poly(x, y):
    # Fit a second order polynomial to each with np.polyfit()
    fit = np.polyfit(y, x, 2)

    return fit


def calc_poly(img_shape, fit):
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    # Calc both polynomials using ploty, left_fit and right_fit
    fitx = fit[0] * ploty ** 2 + fit[1] * ploty + fit[2]

    return ploty, fitx


def search_around_poly(binary_warped, fit, margin=100):
    '''

    :param binary_warped:
    :param fit:
    :param margin: the width of the margin around the previous polynomial to search
    :return:
    '''

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    lane_inds = ((nonzerox > (fit[0] * (nonzeroy ** 2) + fit[1] * nonzeroy + fit[2] - margin)) & (nonzerox < (fit[0] * (nonzeroy ** 2) + fit[1] * nonzeroy + fit[2] + margin)))


    # Again, extract left and right line pixel positions
    x = nonzerox[lane_inds]
    y = nonzeroy[lane_inds]


    return x, y

def draw_poly(img, lane, color = (0, 255, 255), width = 5):

    fit = lane.get_poly()

    if len(img.shape) == 2:
        out_img = np.dstack((img, img, img)) * 255
    else:
        out_img = img.copy()

    window_img = np.zeros_like(out_img)
    ploty, fitx = lane.ploty, lane.fitx
    # Generate x and y values for plotting
    #ploty = np.linspace(0, out_img.shape[0] - 1, out_img.shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    #fitx = fit[0] * ploty ** 2 + fit[1] * ploty + fit[2]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([fitx - width, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([fitx + width,
                                                                    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))


    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), color)
    result = cv2.addWeighted(out_img, 1, window_img, 0.8, 0)

    return result


def draw_lane_pixels(img, lane, color=(255, 0, 0)):

    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    if len(img.shape) == 2:
        out_img = np.dstack((img, img, img)) * 255
    else:
        out_img = img.copy()

    x, y = lane.get_coordinates()

    # Color in left and right line pixels
    out_img[y, x] = color

    return out_img


def draw_lane_area(warped_img, undist_img, left_lane, right_lane, M_inv):
    """
    Returns an image where the inside of the lane has been colored in bright green
    """
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    if len(warped_img.shape) == 2:
        warped_img = np.dstack((warped_img, warped_img, warped_img)) * 255
    else:
        warped_img = warped_img.copy()

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_lane.fitx, left_lane.ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_lane.fitx, right_lane.ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, M_inv, (undist_img.shape[1], undist_img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist_img, 1, newwarp, 0.3, 0)

    return result

def draw_lane_lines(warped_img, undist_img, left_lane, right_lane, M_inv, width):
    """
    Returns an image where the inside of the lane has been colored in bright green
    """
    # Create an image to draw the lines on
    window_img = np.zeros_like(warped_img).astype(np.uint8)
    color_warp = np.dstack((window_img, window_img, window_img))

    if len(warped_img.shape) == 2:
        warped_img = np.dstack((warped_img, warped_img, warped_img)) * 255
    else:
        warped_img = warped_img.copy()

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_lane.fitx - width, left_lane.ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_lane.fitx + width,
                                                                    left_lane.ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_lane.fitx - width, right_lane.ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_lane.fitx + width,
                                                                     right_lane.ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([left_line_pts]), (0, 0, 255))
    cv2.fillPoly(color_warp, np.int_([right_line_pts]), (255, 0, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, M_inv, (undist_img.shape[1], undist_img.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undist_img, 1, newwarp, 0.8, 0)

    return result


def calculate_lane_curvature(left_line, right_line):
    """
    Returns the triple (left_curvature, right_curvature, lane_center_offset), which are all in meters
    """
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 800  # meters per pixel in x dimension
    lane_center_pix = 600

    ploty = left_line.ploty
    y_eval = np.max(ploty)
    # Define conversions in x and y from pixels space to meters

    leftx = left_line.fitx
    rightx = right_line.fitx

    # Fit new polynomials: find x for y in real-world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

    # Now calculate the radii of the curvature
    left_curverad = ((1 + (
                2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])
    right_curverad = ((1 + (
                2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    # Use our computed polynomial to determine the car's center position in image space, then
    left_fit = left_line.best_fit
    right_fit = right_line.best_fit

    center_offset_img_space = (((left_fit[0] * y_eval ** 2 + left_fit[1] * y_eval + left_fit[2]) +
                                (right_fit[0] * y_eval ** 2 + right_fit[1] * y_eval + right_fit[
                                    2])) / 2) - lane_center_pix
    center_offset_real_world_m = center_offset_img_space * xm_per_pix

    # Now our radius of curvature is in meters
    return left_curverad, right_curverad, center_offset_real_world_m
