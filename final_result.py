import cv2
import dlib
import numpy as np


# Load the detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("model/shape_predictor_68_face_landmarks.dat")
# read the image
cap = cv2.VideoCapture("production.mp4")

overlay = cv2.imread('img/pikachu.png', cv2.IMREAD_UNCHANGED)

# overlay function
def overlay_transparent(background_img, img_to_overlay_t, x, y, overlay_size=None):
    try :
        bg_img = background_img.copy()
        # convert 3 channels to 4 channels
        if bg_img.shape[2] == 3:
            bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2BGRA)

        if overlay_size is not None:
            img_to_overlay_t = cv2.resize(img_to_overlay_t.copy(), overlay_size)

        b, g, r, a = cv2.split(img_to_overlay_t)

        mask = cv2.medianBlur(a, 5)

        h, w, _ = img_to_overlay_t.shape
        roi = bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)]

        img1_bg = cv2.bitwise_and(roi.copy(), roi.copy(), mask=cv2.bitwise_not(mask))
        img2_fg = cv2.bitwise_and(img_to_overlay_t, img_to_overlay_t, mask=mask)

        bg_img[int(y-h/2):int(y+h/2), int(x-w/2):int(x+w/2)] = cv2.add(img1_bg, img2_fg)

        # convert 4 channels to 4 channels
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGRA2BGR)

        return bg_img
    except Exception : return background_img


face_roi = []
face_sizes = []
num_size = 0.25

writer = None

# Path to save video
result_path = "result_video.avi"


# loop
while True:
    # read frame buffer from video
    ret, img = cap.read()
    if not ret:
        break

    # resize frame
    img = cv2.resize(img, (int(img.shape[1] * num_size), int(img.shape[0] * num_size)))

    ori = img.copy()

    # find faces
    if len(face_roi) == 0:
        faces = detector(img, 1)
    else:
        roi_img = img[face_roi[0]:face_roi[1], face_roi[2]:face_roi[3]]
        # cv2.imshow('roi', roi_img)
        faces = detector(roi_img)

    # no faces
    if len(faces) == 0:
        print('no faces!')

    # find facial landmarks
    for face in faces:
        if len(face_roi) == 0:
            dlib_shape = predictor(img, face)
            shape_2d = np.array([[p.x, p.y] for p in dlib_shape.parts()])
        else:
            dlib_shape = predictor(roi_img, face)
            shape_2d = np.array([[p.x + face_roi[2], p.y + face_roi[0]] for p in dlib_shape.parts()])

        for s in shape_2d:
            cv2.circle(img, center=tuple(s), radius=1, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        # compute face center
        center_x, center_y = np.mean(shape_2d, axis=0).astype(np.int)

        # compute face boundaries
        min_coords = np.min(shape_2d, axis=0)
        max_coords = np.max(shape_2d, axis=0)

        # draw min, max coords
        cv2.circle(img, center=tuple(min_coords), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(img, center=tuple(max_coords), radius=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

        # compute face size
        face_size = max(max_coords - min_coords)
        face_sizes.append(face_size)
        if len(face_sizes) > 10:
            del face_sizes[0]
        mean_face_size = int(np.mean(face_sizes) * 2.4)

        # compute face roi
        face_roi = np.array([int(min_coords[1] - face_size / 2), int(max_coords[1] + face_size / 2), int(min_coords[0] - face_size / 2), int(max_coords[0] + face_size / 2)])
        face_roi = np.clip(face_roi, 0, 10000)

        # draw overlay on face
        result = overlay_transparent(ori, overlay, center_x - 3 , center_y - 120, overlay_size=(mean_face_size, mean_face_size))

    # visualize
    # show origin
    # cv2.imshow('original', ori)
    # show facial landmark
    # cv2.imshow('facial landmarks', img)
    # frame = cv2.flip(result,0)
    # out.write(frame)

    # save_video = savevideo('save.avi',25.0)
    # save_video.wh_video(result)

    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(result_path, fourcc, 25, (result.shape[1], result.shape[0]), True)

    # 비디오 저장
    if writer is not None:
        writer.write(result)

    #cv2.imshow('result', result)

    if cv2.waitKey(2) == ord('q'):
        #sys.exit(0.1)
        break

cv2.destroyAllWindows()