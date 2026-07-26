import cv2

fs = cv2.FileStorage(
    "camera_intrinsics.yaml",
    cv2.FILE_STORAGE_READ,
)

K = fs.getNode("camera_matrix").mat()
dist = fs.getNode("distortion_coefficients").mat()

fs.release()

img = cv2.imread("calibration_images/image_22.png")

undistorted = cv2.undistort(img, K, dist)

cv2.imshow("original", img)
cv2.imshow("undistorted", undistorted)

cv2.waitKey(0)