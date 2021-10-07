import cv2 as cv
import numpy as np

MIN_MATCH_COUNT = 10

def match(kp1, kp2, features1, features2):
    matcher = cv.BFMatcher()
    matches = matcher.knnMatch(features1, features2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        pts1 = np.float32([kp1[i.queryIdx] for i in good]).reshape(-1, 1, 2)
        pts2 = np.float32([kp2[i.trainIdx] for i in good]).reshape(-1, 1, 2)
        (H, stat) = cv.findHomography(pts1, pts2, cv.RANSAC, 5.0)
        return good, H, stat
    return None
def detect(img):
    detector = cv.SIFT_create()
    (kps, features) = detector.detectAndCompute(img, None)
    kps = np.float32([kp.pt for kp in kps])
    return kps, features
def create_panorama(box):
    (img2, img1) = box
    (kps1, features1) = detect(img1)
    (kps2, features2) = detect(img2)
    M = match(kps1, kps2, features1, features2)
    if M is None:
        return None
    (matches, H, stat) = M
    result = cv.warpPerspective(img1, H, (img1.shape[1]+img2.shape[1], img1.shape[0] + img1.shape[0]//2))
    result[0:img2.shape[0], 0:img2.shape[1]] = img2

    return result

Img1 = cv.imread("Panorama.png")
Img2 = cv.imread("Panorama2.png")
Img3 = cv.imread("Panorama3.png")

border = cv.copyMakeBorder(Img1, 100, 0, 100, 0, cv.BORDER_CONSTANT)

res = create_panorama([Img2, Img3])
resb = create_panorama([border, res])
res = create_panorama([Img1, res])


def trim(frame):
    if not np.sum(frame[3]):
        return trim(frame[1:])

    if not np.sum(frame[-1]):
        return trim(frame[:-2])

    if not np.sum(frame[:, 0]):
        return trim(frame[:, 1:])

    if not np.sum(frame[:, -1]):
        return trim(frame[:, :-2])
    return frame

cv.imshow("Panorama", trim(res))
cv.imwrite("output.jpg", trim(res))
cv.waitKey(0)

cv.destroyAllWindows()
