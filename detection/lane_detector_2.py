import cv2
import numpy as np

class LaneDetector2:
    def __init__(self):
        self.n_windows = 9
        self.margin = 100
        self.minpix = 50

    def preprocess(self, frame):
        # Convert to grayscale and apply Gaussian blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Sobel operator or binary thresholding
        _, binary = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)
        return binary

    def perspective_transform(self, image):
        height, width = image.shape[:2]
        src = np.float32([[width*0.45, height*0.65],
                          [width*0.55, height*0.65],
                          [width*0.1, height],
                          [width*0.9, height]])
        dst = np.float32([[width*0.2, 0],
                          [width*0.8, 0],
                          [width*0.2, height],
                          [width*0.8, height]])

        matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(image, matrix, (width, height))
        return warped

    def sliding_window(self, binary_warped):
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))

        midpoint = int(histogram.shape[0] // 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        window_height = int(binary_warped.shape[0] // self.n_windows)
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        leftx_current = leftx_base
        rightx_current = rightx_base

        left_lane_inds = []
        right_lane_inds = []

        for window in range(self.n_windows):
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height

            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > self.minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        left_fit = np.polyfit(lefty, leftx, 2) if len(leftx) > 0 else None
        right_fit = np.polyfit(righty, rightx, 2) if len(rightx) > 0 else None

        return left_fit, right_fit, out_img

    def draw_lane(self, frame, binary_warped, left_fit, right_fit):
        ploty = np.linspace(0, frame.shape[0] - 1, frame.shape[0])
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        if left_fit is not None and right_fit is not None:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

            pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
            pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
            pts = np.hstack((pts_left, pts_right))

            cv2.fillPoly(color_warp, [np.int32(pts)], (0, 255, 0))

        height, width = frame.shape[:2]
        src = np.float32([[width*0.45, height*0.65],
                          [width*0.55, height*0.65],
                          [width*0.1, height],
                          [width*0.9, height]])
        dst = np.float32([[width*0.2, 0],
                          [width*0.8, 0],
                          [width*0.2, height],
                          [width*0.8, height]])
        Minv = cv2.getPerspectiveTransform(dst, src)
        unwarped = cv2.warpPerspective(color_warp, Minv, (width, height))

        result = cv2.addWeighted(frame, 1, unwarped, 0.3, 0)
        return result
