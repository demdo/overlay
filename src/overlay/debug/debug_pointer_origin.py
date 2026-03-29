import cv2
import numpy as np
import pyrealsense2 as rs


# ============================================================
# ArUco Setup (WICHTIG: gleiche Config wie dein Tool!)
# ============================================================

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

TARGET_ID = 8


# ============================================================
# RealSense Setup (wie bei dir)
# ============================================================

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)

pipeline.start(config)


print("Tracking Marker ID 8 (Board Origin)")
print("ESC / q to quit")

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        img = np.asanyarray(color_frame.get_data())
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            ids = ids.flatten()

            for i, marker_id in enumerate(ids):
                if marker_id == TARGET_ID:

                    c = corners[i].reshape(4, 2)

                    # Mittelpunkt
                    center = np.mean(c, axis=0)

                    # Zeichnen
                    cv2.aruco.drawDetectedMarkers(img, corners, ids)

                    cv2.circle(
                        img,
                        (int(center[0]), int(center[1])),
                        8,
                        (0, 0, 255),
                        -1
                    )

                    cv2.putText(
                        img,
                        "ORIGIN (ID 8)",
                        (int(center[0]) + 10, int(center[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                    # optional: pixel coords anzeigen
                    cv2.putText(
                        img,
                        f"({center[0]:.1f}, {center[1]:.1f})",
                        (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                    )

                    break

        cv2.imshow("Board Origin Debug", img)

        key = cv2.waitKey(1)
        if key in [27, ord("q")]:
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()