import openpifpaf
import cv2
import numpy as np

img = cv2.imread("images/openpifpaf_sample.jpeg")

predicter = openpifpaf.Predictor(checkpoint="shufflenetv2k16")
# print(predicter.numpy_image(img))
predictions, gt_anns, meta = predicter.numpy_image(img)
index = predictions[0].keypoints.index("right_hip")
# print(index)
# print(predictions[0].keypoints[index])
# print(predictions[0].data[index])
# print(predictions[0].keypoints)

for pred in predictions:
    # print("==========================")
    for pt in pred.data[:, :2].astype("int"):
        # print(pt)
        cv2.circle(img, center=pt, radius=5, color=(255, 255, 0), thickness=-1)
    left_shoulder = pred.data[pred.keypoints.index("left_shoulder")][:2]
    right_shoulder = pred.data[pred.keypoints.index("right_shoulder")][:2]
    left_hip = pred.data[pred.keypoints.index("left_hip")][:2]
    right_hip = pred.data[pred.keypoints.index("right_hip")][:2]
    # print("left_shoulder, right_shoulder, left_hip, right_hip")
    # print(left_shoulder, right_shoulder, left_hip, right_hip)
    center_shoulder = np.mean([left_shoulder, right_shoulder], 0, np.int16)[:2]
    center_hip = np.mean([left_hip, right_hip], 0, np.int16)[:2]
    # print("center_shoulder, center_hip")
    # print(center_shoulder, center_hip)
    cv2.circle(img, center=center_shoulder, radius=5,
               color=(255, 0, 0), thickness=-1)
    cv2.circle(img, center=center_hip, radius=5,
               color=(255, 0, 0), thickness=-1)
    cv2.line(img, pt1=center_shoulder, pt2=center_hip,
             color=(255, 0, 0), thickness=3)
    # 角度
    vec = (center_shoulder - center_hip) * [1, -1]
    angle = np.arctan2(vec[0], vec[1]) * 180 / np.pi
    # print("angle")
    # print(angle)
    cv2.putText(
        img,
        text=f"{angle:.2f}",
        org=(center_shoulder[0], center_shoulder[1]),
        fontFace=cv2.FONT_HERSHEY_PLAIN,
        fontScale=1,
        color=(255, 0, 0),
        thickness=2,
    )

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
