import vimba
import cv2
import numpy as np
from calibration_routine import calibrate_camera
from cv2 import aruco
import open3d as o3d

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((5*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:5].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
mtx = np.loadtxt('mtx.txt', dtype=float)
dist = np.loadtxt('dist.txt', dtype=float)

aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)
board = aruco.CharucoBoard_create(7, 5, 0.03, 0.015, aruco_dict)
imboard = board.draw((1000, 1000))
cv2.imwrite('Charuco_board.png', imboard)
cv2.namedWindow("charuco")
cv2.imshow("charuco", imboard)
arucoParams = aruco.DetectorParameters_create()
last_img = None
rvec=np.array([])
tvec=np.array([])
pcd = o3d.io.read_point_cloud('./3d_objects/stanford_bunny.ply')
cord_change_mat = np.array([[1., 0., 0.], [0, 1., 0.], [0., 0., -1.]], dtype=np.float32)
pcd.rotate(cord_change_mat)
pcd.scale(2, center=pcd.get_center())
pcd.translate(-pcd.get_center())
pcd.translate([0,0,-0.05])
pts3D = np.asarray(pcd.points)
# pts3D = pts3D.dot(cord_change_mat.T)

with vimba.Vimba.get_instance() as vmb:
    cams = vmb.get_all_cameras()
    with cams[0] as cam:
        # Adjust to the Bayer format you want to use
        cam.set_pixel_format(vimba.PixelFormat.BayerRG8)
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
        cv2.namedWindow("output_uncal", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
        while 1:
            # Record single frame for this example
            frame = cam.get_frame()
            # Get the raw image data as numpy array from the frame
            image = frame.as_numpy_ndarray()
            # Use opencv to convert raw Bayer image to RGB image
            img = cv2.cvtColor(image, cv2.COLOR_BAYER_RG2RGB)
            #dst = cv2.undistort(img, mtx, dist, None)
            dst=img
            dst_gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
            if last_img is not img:
                corners, ids, rejectedImgPoints = aruco.detectMarkers(dst, aruco_dict,
                                                                      parameters=arucoParams)
                aruco.refineDetectedMarkers(dst_gray, board, corners, ids, rejectedImgPoints)
                if len(corners) > 0:
                    charucoretval, charucoCorners, charucoIds = aruco.interpolateCornersCharuco(corners, ids,
                                                                                                dst_gray,
                                                                                                board)
                    im_with_charuco_board = aruco.drawDetectedCornersCharuco(dst, charucoCorners, charucoIds,
                                                                             (0, 255, 0))
                    retval, rvec, tvec = aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, mtx, dist, rvec, tvec)  # posture estimation from a charuco board
                    if retval == True:
                        dst = cv2.drawFrameAxes(dst, mtx, dist, rvec, tvec, 0.1)  # axis length 100 can be changed according to your requirement
                        img_points, _ = cv2.projectPoints(pts3D, rvec, tvec, mtx, dist)
                        for point in img_points:
                            p1, p2 = int(point[0][0]), int(point[0][1])
                            if p2 >= dst.shape[0] or p1 >= dst.shape[1] or p1<0 or p2<0:
                                continue
                            else:
                                dst[p2, p1] = (255, 255, 255)

                cv2.imshow("output", dst)
                # cv2.imshow("output_uncal", img_points)
                cv2.waitKey(100)


cv2.destroyAllWindows()