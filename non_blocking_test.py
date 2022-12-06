import open3d as o3d
import numpy as np
import vimba
import cv2
import numpy as np
from calibration_routine import calibrate_camera
from cv2 import aruco
import open3d as o3d



if __name__ == "__main__":
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((5 * 8, 3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:5].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
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
    rvec = np.array([])
    tvec = np.array([])

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    pcd_data = o3d.data.DemoICPPointClouds()
    source_raw = o3d.io.read_point_cloud(pcd_data.paths[0])
    target_raw = o3d.io.read_point_cloud(pcd_data.paths[1])

    source = o3d.io.read_point_cloud('./3d_objects/stanford_bunny.ply')
    source.translate(-source.get_center())
    source.translate([0, 0, -0.05])
    pts3D = np.asarray(source.points)

    flip_transform = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    source.transform(flip_transform)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(source)
    x = o3d.cpu.pybind.camera.PinholeCameraIntrinsic(2064, 1544, 1109, 1118, 1012, 830)
    params = o3d.camera.PinholeCameraParameters()
    params.intrinsic = x
    params.extrinsic = np.eye(4,4)
    # vis.ViewControl.convert_from_pinhole_camera_parameters(x)
    view_control = vis.get_view_control()
    view_control.convert_from_pinhole_camera_parameters(params)
    view_control.set_lookat([0, 0, 1])
    rotation_mat = None
    old_rvec = None
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
                # dst = cv2.undistort(img, mtx, dist, None)
                dst = img
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
                        retval, rvec, tvec = aruco.estimatePoseCharucoBoard(charucoCorners, charucoIds, board, mtx,
                                                                            dist, rvec,
                                                                            tvec)  # posture estimation from a charuco board
                        if retval == True:

                            if old_rvec is None:
                                rotation_mat, _ = cv2.Rodrigues(rvec)
                                old_rvec = rvec.copy()
                            else:
                                rotation_mat, _ = cv2.Rodrigues(rvec - old_rvec)
                                old_rvec = rvec.copy()
                            source.rotate(rotation_mat)
                            temp_tvec = tvec.copy()
                            # temp_tvec[2] = -1*temp_tvec[2]
                            print(temp_tvec)
                            source.translate(temp_tvec, relative=False)
                vis.update_geometry(source)
                vis.poll_events()
                vis.update_renderer()

            vis.destroy_window()
            o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)