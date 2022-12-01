import vimba
import cv2
import numpy as np

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((5*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:5].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
with vimba.Vimba.get_instance() as vmb:
    cams = vmb.get_all_cameras()
    with cams[0] as cam:
        # Adjust to the Bayer format you want to use
        cam.set_pixel_format(vimba.PixelFormat.BayerRG8)
        while 1:
            # Record single frame for this example
            frame = cam.get_frame()
            # Get the raw image data as numpy array from the frame
            image = frame.as_numpy_ndarray()
            # Use opencv to convert raw Bayer image to RGB image
            img  = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2RGB)
            cv2.imwrite('uncalibrated.png', img)
            img2 = img
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (8, 5), flags=None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                # Draw and display the corners
                cv2.drawChessboardCorners(img, (8, 5), corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey(0)
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

                h, w = img2.shape[:2]
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

                # undistort
                dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

                # crop the image
                # x, y, w, h = roi
                # dst = dst[y:y + h, x:x + w]
                cv2.imwrite('calibresult.png', dst)


cv2.destroyAllWindows()