# -*- coding: utf-8 -*-
# import the necessary packages
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import math
import tkinter
import tkinter.messagebox
import time
from threading import Thread

# (UVW)
object_pts = np.float32(
    [
        [6.825897, 6.760612, 4.402142],  # 33左眉左上角
        [1.330353, 7.122144, 6.903745],  # 29左眉右角
        [-1.330353, 7.122144, 6.903745],  # 34右眉左角
        [-6.825897, 6.760612, 4.402142],  # 38右眉右上角
        [5.311432, 5.485328, 3.987654],  # 13左眼左上角
        [1.789930, 5.393625, 4.413414],  # 17左眼右上角
        [-1.789930, 5.393625, 4.413414],  # 25右眼左上角
        [-5.311432, 5.485328, 3.987654],  # 21右眼右上角
        [2.005628, 1.409845, 6.165652],  # 55鼻子左上角
        [-2.005628, 1.409845, 6.165652],  # 49鼻子右上角
        [2.774015, -2.080775, 5.048531],  # 43嘴左上角
        [-2.774015, -2.080775, 5.048531],  # 39嘴右上角
        [0.000000, -3.116408, 6.097667],  # 45嘴中央下角
        [0.000000, -7.415691, 4.070434],
    ]
)  # 6下巴角

# 相机坐标系(XYZ)：添加相机内参
K = [
    6.5308391993466671e002,
    0.0,
    3.1950000000000000e002,
    0.0,
    6.5308391993466671e002,
    2.3950000000000000e002,
    0.0,
    0.0,
    1.0,
]  # 等价于矩阵[fx, 0, cx; 0, fy, cy; 0, 0, 1]
# 图像中心坐标系(uv)：相机畸变参数[k1, k2, p1, p2, k3]
D = [
    7.0834633684407095e-002,
    6.9140193737175351e-002,
    0.0,
    0.0,
    -1.3073460323689292e000,
]

cam_matrix = np.array(K).reshape(3, 3).astype(np.float32)
dist_coeffs = np.array(D).reshape(5, 1).astype(np.float32)
reprojectsrc = np.float32(
    [
        [10.0, 10.0, 10.0],
        [10.0, 10.0, -10.0],
        [10.0, -10.0, -10.0],
        [10.0, -10.0, 10.0],
        [-10.0, 10.0, 10.0],
        [-10.0, 10.0, -10.0],
        [-10.0, -10.0, -10.0],
        [-10.0, -10.0, 10.0],
    ]
)

line_pairs = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
]


def cal_ang(point_1, point_2, point_3):
    a = math.sqrt(
        (point_2[0] - point_3[0]) * (point_2[0] - point_3[0]) + (point_2[1] - point_3[1]) * (point_2[1] - point_3[1]))
    b = math.sqrt(
        (point_1[0] - point_3[0]) * (point_1[0] - point_3[0]) + (point_1[1] - point_3[1]) * (point_1[1] - point_3[1]))
    c = math.sqrt(
        (point_1[0] - point_2[0]) * (point_1[0] - point_2[0]) + (point_1[1] - point_2[1]) * (point_1[1] - point_2[1]))
    B = math.degrees(math.acos((b * b - a * a - c * c) / (-2 * a * c)))
    return B


def cal_distance(pt1, pt2):
    pt3 = pt1 - pt2
    return round(math.hypot(pt3[0], pt3[1]), 5)


def get_head_pose(shape):  # 头部姿态估计
    # （像素坐标集合）填写2D参考点，注释遵循https://ibug.doc.ic.ac.uk/resources/300-W/
    # 17左眉左上角/21左眉右角/22右眉左上角/26右眉右上角/36左眼左上角/39左眼右上角/42右眼左上角/
    # 45右眼右上角/31鼻子左上角/35鼻子右上角/48左上角/54嘴右上角/57嘴中央下角/8下巴角
    image_pts = np.float32(
        [
            shape[17],
            shape[21],
            shape[22],
            shape[26],
            shape[36],
            shape[39],
            shape[42],
            shape[45],
            shape[31],
            shape[35],
            shape[48],
            shape[54],
            shape[57],
            shape[8],
        ]
    )

    _, rotation_vec, translation_vec = cv2.solvePnP(
        object_pts, image_pts, cam_matrix, dist_coeffs
    )
    reprojectdst, _ = cv2.projectPoints(
        reprojectsrc, rotation_vec, translation_vec, cam_matrix, dist_coeffs
    )
    # change to int32(xy)
    reprojectdst = reprojectdst.astype(np.int32)
    reprojectdst = tuple(map(tuple, reprojectdst.reshape(8, 2)))  # 以8行2列显示
    # calc euler angle
    # https://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#decomposeprojectionmatrix
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, translation_vec))
    # decomposeProjectionMatrix
    _, _, _, _, _, _, euler_angle = cv2.decomposeProjectionMatrix(pose_mat)
    pitch, yaw, roll = [math.radians(_) for _ in euler_angle]
    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))
    print("pitch:{}, yaw:{}, roll:{}".format(pitch, yaw, roll))

    return reprojectdst, euler_angle


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


def mouth_aspect_ratio(mouth):
    A = np.linalg.norm(mouth[2] - mouth[9])  # 51, 59
    B = np.linalg.norm(mouth[4] - mouth[7])  # 53, 57
    C = np.linalg.norm(mouth[0] - mouth[6])  # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar


def detect_all():
    EYE_AR_THRESH = 0.2
    EYE_AR_CONSEC_FRAMES = 3
    MAR_THRESH = 0.5
    MOUTH_AR_CONSEC_FRAMES = 3
    HAR_THRESH = 0.3
    NOD_AR_CONSEC_FRAMES = 3
    COUNTER = 0
    TOTAL = 0
    mCOUNTER = 0
    mTOTAL = 0
    hCOUNTER = 0
    hTOTAL = 0

    total_sleepy_cnt = 0
    total_confused_cnt = 0
    eyeBrow_dis = 0

    print("[INFO] loading facial landmark predictor...")
    # dlib.get_frontal_face_detector()
    detector = dlib.get_frontal_face_detector()
    # dlib.shape_predictor
    predictor = dlib.shape_predictor(
        "/Users/xuyuansmacbook/Desktop/cityhack/shape_predictor_68_face_landmarks.dat"
    )

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    # open cam
    cap = cv2.VideoCapture(0)
    time_start = time.time()
    count = 0
    while cap.isOpened():
        # read img and turn grey
        ret, frame = cap.read()
        frame = imutils.resize(frame, width=720)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # find face
        rects = detector(gray, 0)
        if len(rects) > 0:

            for rect in rects:
                shape = predictor(gray, rect)
                # turn array
                shape = face_utils.shape_to_np(shape)
                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                mouth = shape[mStart:mEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0
                # yawning
                mar = mouth_aspect_ratio(mouth)
                leftEyeHull = cv2.convexHull(leftEye)
                rightEyeHull = cv2.convexHull(rightEye)
                cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
                mouthHull = cv2.convexHull(mouth)
                cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
                left = rect.left()
                top = rect.top()
                right = rect.right()
                bottom = rect.bottom()
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 1)
                if ear < EYE_AR_THRESH:
                    COUNTER += 1
                else:
                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        TOTAL += 1
                    COUNTER = 0
                cv2.putText(
                    frame,
                    "Faces: {}".format(len(rects)),
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "COUNTER: {}".format(COUNTER),
                    (150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "EAR: {:.2f}".format(ear),
                    (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "Blinks: {}".format(TOTAL),
                    (450, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )
                if mar > MAR_THRESH:
                    mCOUNTER += 1
                    cv2.putText(
                        frame,
                        "Yawning!",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                else:
                    if mCOUNTER >= MOUTH_AR_CONSEC_FRAMES:
                        mTOTAL += 1
                    mCOUNTER = 0
                cv2.putText(
                    frame,
                    "COUNTER: {}".format(mCOUNTER),
                    (150, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "MAR: {:.2f}".format(mar),
                    (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "Yawning: {}".format(mTOTAL),
                    (450, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )

                reprojectdst, euler_angle = get_head_pose(shape)
                har = euler_angle[0, 0]
                if har > HAR_THRESH:
                    hCOUNTER += 1
                else:
                    if hCOUNTER >= NOD_AR_CONSEC_FRAMES:
                        hTOTAL += 1
                    hCOUNTER = 0

                for start, end in line_pairs:
                    cv2.line(frame, reprojectdst[start], reprojectdst[end], (0, 0, 255))

                cv2.putText(
                    frame,
                    "X: " + "{:7.2f}".format(euler_angle[0, 0]),
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 0),
                    thickness=2,
                )  # GREEN
                cv2.putText(
                    frame,
                    "Y: " + "{:7.2f}".format(euler_angle[1, 0]),
                    (150, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 0, 0),
                    thickness=2,
                )  # BLUE
                cv2.putText(
                    frame,
                    "Z: " + "{:7.2f}".format(euler_angle[2, 0]),
                    (300, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    thickness=2,
                )  # RED
                cv2.putText(
                    frame,
                    "Nod: {}".format(hTOTAL),
                    (450, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 0),
                    2,
                )

                for x, y in shape:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                print(
                    "mouth width/height:{:.2f} ".format(mar) + "\tmouth open?：" + str([False, True][mar > MAR_THRESH])
                )
                print("eye width/height:{:.2f} ".format(ear) + "\twink?：" + str([False, True][COUNTER >= 1]))

                if TOTAL >= 50 or mTOTAL >= 15 or hTOTAL >= 15:
                    cv2.putText(
                        frame, "SLEEP!!!", (100, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3
                    )
                    total_sleepy_cnt += 1
                    TOTAL = 0
                    mTOTAL = 0
                    hTOTAL = 0
                # facial expression cal
                # mouth open and 皱眉
                eyeBrow_dis_new = cal_distance(shape[21], shape[22])
                count += 1
                eyeBrow_dis += eyeBrow_dis_new
                eyeBrow_dis_average = eyeBrow_dis/count
                if eyeBrow_dis_new < eyeBrow_dis_average:
                    if total_confused_cnt > 300:
                        cv2.putText(
                            frame, "Confused", (left, bottom), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3
                        )
                    total_confused_cnt += 1
            cv2.putText(
                frame, "sleepy_cnt: {}".format(total_sleepy_cnt), (0, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255),
                3
            )
            cv2.putText(
                frame, "confused_cnt: {}".format(total_confused_cnt), (0, 330), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255), 3
            )
            time_end = time.time()
            cv2.putText(
                frame, "time: {}".format(time_end-time_start), (0, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255), 3
            )
            if total_confused_cnt < 300:
                temp_confused_cnt = 0
            else:
                temp_confused_cnt = total_confused_cnt
            total_positive_score = 100*(total_sleepy_cnt*0.35+temp_confused_cnt*0.02)/(time_end-time_start)
            cv2.putText(
                frame, "positive_score: {}".format(total_positive_score), (0, 400), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 255), 3
            )
        # show with opencv
        cv2.imshow("Frame", frame)
        # if the `q` key was pressed, break from the loop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    # release camera
    cap.release()
    # do a bit of cleanup
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detect_all()
