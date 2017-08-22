import numpy as np
import cv2
import glob
import matplotlib
#matplotlib.use('qt5agg')
import matplotlib.pyplot as plt


# %matplotlib qt5
#%matplotlib qt

def grayscale(img):
    """Applies the Grayscale transform"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def calibrate(images):
    """Computes the camera calibration using chessboard images"""
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.


    # Step through the list and search for chessboard corners
    shape = []
    for fname in images:
        print("processing image: {}".format(fname))
        img = cv2.imread(fname)
        shape = img.shape
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

    retval, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape[0:2], None, None)
    return cameraMatrix, distCoeffs


def undistort_image(img, mtx, dist):
    """Apply a distortion correction to raw image"""
    import matplotlib.image as mpimg
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return dst

def hls_transform(img):
    """Applies the hsl transform"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

def sobel_operator(img, dir='x'):
    gray = grayscale(img)
    if (dir == 'x'):
        return cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        return cv2.Sobel(gray, cv2.CV_64F, 0, 1)

def sobel_scale(img_sobel):
    """Absolute the derivative to accentuate lines away from horizontal/vertical??"""
    abs_sobel = np.absolute(img_sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    return scaled_sobel

def select_yellow(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower = np.array([20,60,60])
    upper = np.array([38,174, 250])
    mask = cv2.inRange(hsv, lower, upper)
    return mask

def select_white(img):
    lower = np.array([202,202,202])
    upper = np.array([255,255,255])
    mask = cv2.inRange(img, lower, upper)
    return mask

def apply_mask(img, mask):
    result = cv2.bitwise_and(img,img,mask=mask)
    return result

def threshold_image(img, thresh=(20, 100)):
    s_binary = np.zeros_like(img)
    s_binary[(img >= thresh[0]) & (img <= thresh[1])] = 1
    return s_binary

def binary_image_transform_yellow_white_lane_lines(img):
    yellow_mask = select_yellow(img)
    yellow_image = apply_mask(img, yellow_mask)
    ret,yellow_binary = cv2.threshold(grayscale(yellow_image),127,255,cv2.THRESH_BINARY)

    white_mask = select_white(img)
    white_image = apply_mask(img, white_mask)
    ret,white_binary = cv2.threshold(grayscale(white_image),127,255,cv2.THRESH_BINARY)

    combined_binary = np.zeros_like(white_binary)
    combined_binary[(yellow_binary > 0) | (white_binary > 0)] = 1
    #     plot_2_images(img, combined_binary)
    return combined_binary

def plot_2_images(img1, img2, title1='original',title2='processed'):
    import matplotlib.pyplot as plt

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(48, 18))
    f.tight_layout()
    ax1.imshow(img1)
    ax1.set_title(title1, fontsize=50)
    ax2.imshow(img2, cmap='gray')
    ax2.set_title(title2, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)


def plot_image_title(img, title='', text=''):

    font = {'family': 'serif',
            'color':  'white',
            'weight': 'normal',
            'size': 28,
            }

    f, ax = plt.subplots(1, 1, figsize=(48, 18))
    f.tight_layout()
    ax.imshow(img)
    plt.text(2, 75.65, text, fontdict=font)
    ax.set_title(title, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
#     plt.savefig(output_file)

def binary_image_transform(img):
    """Uses gradients to create a binary image"""

    # Threshold x gradient
    sobel_x = sobel_operator(img, dir='x')
    s_x_binary = threshold_image(sobel_scale(sobel_x), thresh=(20, 100))

    # Threshold color channel
    hls = hls_transform(img)
    s_channel = hls[:,:,2]
    s_binary = threshold_image(s_channel, thresh=(170, 255))

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(s_x_binary)
    combined_binary[(s_binary == 1) | (s_x_binary == 1)] = 1
    return combined_binary

def unwarp(img, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped, M

def hist(img):
    return np.sum(img[img.shape[0]//2:,:], axis=0)


def polyfit(img, y, x, order=2):
    """Fit polynomial"""
    fit = np.poly1d(np.polyfit(y, x, order))

    #y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    fit_x = fit(ploty)
    return fit_x


def detect_lane_lines(binary_warped, debug = False,nwindows = 9, margin = 100, minpix = 50):
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)

    histogram = hist(binary_warped)

    out_img = None
    if (debug==True):
        # For debugging, an output image to visualize the result
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint


    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base


    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):

        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        if (debug==True):
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)

            # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if (len(rightx)==0 or len(leftx)==0):
        return [], [], [], []

    #     print("len(rightx) {}".format(len(rightx)))
    #     print("len(leftx) {}".format(len(leftx)))
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

    left_fitx = polyfit(binary_warped, lefty, leftx, 2)
    right_fitx = polyfit(binary_warped, righty, rightx, 2)

    if (debug==True):
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
    return left_fit, right_fit, left_fitx, right_fitx

def detect_lane_lines_subsequent_images(binary_warped, left_fit, right_fit, debug=False, margin = 100):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    #TODO improve this
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fitx = polyfit(binary_warped, lefty, leftx, 2)
    right_fitx = polyfit(binary_warped, righty, rightx, 2)


    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )

    if (debug==True):
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

    return left_fit, right_fit, left_fitx, right_fitx


def curvature(binary_warped, left_fitx, right_fitx):
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    y_eval = np.max(ploty)

    leftx = left_fitx
    rightx = right_fitx

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    # Now our radius of curvature is in meters
    center_of_lanes = ((right_fitx[-1] - left_fitx[-1]) //2 + left_fitx[-1]) * xm_per_pix
    center_of_car = (binary_warped.shape[1] // 2) * xm_per_pix
    return left_curverad, right_curverad, (center_of_lanes - center_of_car)

def warp_detected_lines_onto_original(original, warped, left_fitx, right_fitx, Minv):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (original.shape[1], original.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(original, 1, newwarp, 0.3, 0)
    #     plt.imshow(result)
    return result

## Compute the camera calibration using chessboard images
images = glob.glob('../camera_cal/calibration*.jpg')
cameraMatrix, distCoeffs = calibrate(images)

## Apply a distortion correction to a raw images
img = mpimg.imread('../test_images/test5.jpg')
undist = undistort_image(img, cameraMatrix, distCoeffs)
plot_2_images(img, undist)

## Create a thresholded binary image.
binary_image = binary_image_transform_yellow_white_lane_lines(undist)
plot_2_images(undist, binary_image)


## Apply a perspective transform to rectify binary image ("birds-eye view").
img_size = (binary_image.shape[1], binary_image.shape[0])
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
     [((img_size[0] / 6) - 10), img_size[1]],
     [(img_size[0] * 5 / 6) + 40, img_size[1]],
     [(img_size[0] / 2 + 65), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
     [(img_size[0] / 4), img_size[1]],
     [(img_size[0] * 3 / 4), img_size[1]],
     [(img_size[0] * 3 / 4), 0]])
warped, M = unwarp(binary_image, src, dst)
Minv = cv2.getPerspectiveTransform(dst, src)

print(src)
# plot_2_images(undist, warped)

warped_undist, M2 = unwarp(undist, src, dst)

src2 = np.int32(src.reshape((-1,1,2)))
dst2 = np.int32(dst.reshape((-1,1,2)))

cv2.polylines(undist,[src2],True,(255,0,0))
cv2.polylines(warped_undist,[dst2],True,(255,0,0))

plot_2_images(undist, warped_undist)

## Detect lane pixels and fit to find the lane boundary.
left_fit, right_fit, left_fitx, right_fitx = detect_lane_lines(warped, debug=True)
left_fit, right_fit, left_fitx, right_fitx = detect_lane_lines_subsequent_images(warped, left_fit, right_fit, debug=True)

## Determine the curvature of the lane and vehicle position with respect to center.
left_curverad, right_curverad, diff_center = curvature(warped, left_fitx, right_fitx)
print(left_curverad, 'm', right_curverad, 'm', diff_center)

## Warp the detected lane boundaries back onto the original image.
result = warp_detected_lines_onto_original(undist, warped, left_fitx, right_fitx, Minv)
plot_2_images(img, result)


## Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

# title = "Radius of curvature = {:f}(m)\nVehicle is {:f}m {!s} of center ".format(np.mean(left_curverad,right_curverad), np.abs(diff_center), 'left')
text = 'Radius of curvature = {:f}(m)\nVehicle is {:f}m {!s} of center'.format(np.mean([left_curverad,right_curverad]), np.abs(diff_center), 'left' if diff_center > 0 else 'right')
plot_image_title(result, '', text)

## Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        #loss count
        self.loss_count = 0
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients of the last n fits of the line
        self.recent_fit = []
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
    #update the stats
    def update(self, fit, fitx, radius_of_curvature, line_base_pos, warped):
        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
        n = len(self.recent_xfitted)

        if (n > 10):
            removed_fitx = self.recent_xfitted.pop(0)
            removed_fit = self.recent_fit.pop(0)

        self.recent_xfitted.append(fitx)
        self.bestx = np.mean(self.recent_xfitted, axis=0,keepdims=True)
        self.recent_fit.append(fit)
        self.best_fit = np.mean(self.recent_fit)
        self.radius_of_curvature = radius_of_curvature
        self.line_base_pos = line_base_pos
        self.diffs = self.current_fit - fit
        self.current_fit = fit
        self.allx = fitx
        self.ally = ploty
        self.detected = True
        self.loss_count = np.max([self.loss_count-1, 0])
    def reset():
        self.detected = False
        self.recent_xfitted = []
        self.bestx = None
        self.best_fit = None
        self.current_fit = [np.array([False])]
        self.radius_of_curvature = None
        self.line_base_pos = None
        self.diffs = np.array([0,0,0], dtype='float')
        self.allx = None
        self.ally = None
        self.loss_count=0



## Pipeline
def sanity_check(leftLine, rightLine, left_fit, right_fit, left_fitx, right_fitx, left_curverad, right_curverad, diff_center):

    #Checking that lines have similar curvature
    if ( (left_curverad-right_curverad) / right_curverad > 2):
        print("left_curverad, right_curverad")
        print(left_curverad, right_curverad)
        return False

    #Checking that lines are separated by approximately the right distance horizontally
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    if ( ((right_fitx[-1]-left_fitx[-1]) * xm_per_pix - 3.7) > 5e-2):
        print("right_fitx[-1], left_fitx[-1")
        print(right_fitx[-1], left_fitx[-1])
        return False

    #     #Checking that lines are roughly parallel
    upper = right_fitx[0] - left_fitx[0]
    lower = right_fitx[-1] - left_fitx[-1]
    if ( (upper-lower) / lower > 11e-2):
        print("upper, lower")
        print(upper, lower)
        return False
    #     print('great, passed sanity checks!')

    return True


def sanity_check_update_lines(left_fit, right_fit, left_fitx, right_fitx, warped, leftLine, rightLine):
    if (len(left_fit) == 0):
        leftLine.detected = False
        leftLine.loss_count += 1

        rightLine.detected = False
        rightLine.loss_count += 1

        return 0., 0., 0.
    left_curverad, right_curverad, diff_center = curvature(warped, left_fitx, right_fitx)

    if (sanity_check(leftLine, rightLine, left_fit, right_fit, left_fitx, right_fitx, left_curverad, right_curverad, diff_center)):
        leftLine.update(left_fit, left_fitx, left_curverad, diff_center, warped)
        rightLine.update(right_fit, right_fitx, right_curverad, diff_center, warped)
    else:
        leftLine.detected = False
        leftLine.loss_count += 1

        rightLine.detected = False
        rightLine.loss_count += 1

    return left_curverad, right_curverad, diff_center


def pipeline(img, simple=True):
    """
    1) Sanity Check
    2) Look-Ahead Filter
    3) Reset
    4) Smoothing
    5) Drawing
    """
    undist = undistort_image(img, cameraMatrix, distCoeffs)
    binary_image = binary_image_transform_yellow_white_lane_lines(undist)
    warped, M = unwarp(binary_image, src, dst)

    if (simple):
        left_fit, right_fit, left_fitx, right_fitx = detect_lane_lines(warped)
        left_curverad, right_curverad, diff_center = sanity_check_update_lines(left_fit, right_fit, left_fitx, right_fitx, warped, leftLine, rightLine)
    else:
        if (len(leftLine.recent_xfitted) > 0 & leftLine.loss_count < 2 & len(rightLine.recent_xfitted) > 0  & rightLine.loss_count < 5):
            left_fit, right_fit, left_fitx, right_fitx = detect_lane_lines_subsequent_images(warped, leftLine.best_fit, rightLine.best_fit)
            left_curverad, right_curverad, diff_center = sanity_check_update_lines(left_fit, right_fit, left_fitx, right_fitx, warped, leftLine, rightLine)
        else:
            print('resetting....')
            leftLine.reset
            rightLine.reset
            left_fit, right_fit, left_fitx, right_fitx = detect_lane_lines(warped)
            left_curverad, right_curverad, diff_center = sanity_check_update_lines(left_fit, right_fit, left_fitx, right_fitx, warped, leftLine, rightLine)

    if (left_curverad == 0. or right_curverad == 0.):
        return img
    result = warp_detected_lines_onto_original(undist, warped, leftLine.bestx, rightLine.bestx, Minv)
    text1 = 'Radius of curvature = {:f}(m)'.format(np.mean([left_curverad,right_curverad]))
    text2 = 'Vehicle is {:f}m {!s} of center'.format(np.abs(diff_center), 'left' if diff_center > 0 else 'right')
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result,text1,(10,50), font, 1,(255,255,255),2,cv2.LINE_AA)
    cv2.putText(result,text2,(10,95), font, 1,(255,255,255),2,cv2.LINE_AA)

    return result

import os
test_image_file_names = os.listdir("../test_images/")
leftLine = Line()
rightLine = Line()
for image_file_name in test_image_file_names:
    print(image_file_name)
    image = mpimg.imread("../test_images/" + image_file_name)
    result = pipeline(image, simple=True)
    mpimg.imsave("../output_images/" + image_file_name, result)








