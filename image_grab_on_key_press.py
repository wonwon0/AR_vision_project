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
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
    with cams[0] as cam:
        cam.set_pixel_format(vimba.PixelFormat.BayerRG8)
        image_number = 0
        while 1:
            # Adjust to the Bayer format you want to use
            # Record single frame for this example
            frame = cam.get_frame()
            # Get the raw image data as numpy array from the frame
            image = frame.as_numpy_ndarray()
            img = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2RGB)
            cv2.imshow('output',img)
            k = cv2.waitKey(1)
            if k == ord('g'):
                # Use opencv to convert raw Bayer image to RGB image
                cv2.imwrite('./calibration_images/image'+str(image_number)+'.png', img)
                print('image' + str(image_number) + ' saved')
                image_number+=1



