import cv2
import numpy as np

def findSameAngle(points, errorRate = 0.2, minR = 10):
    편차 = points - points.mean((0,1)).reshape(1, 1, 2)
    # N , 1, 2
    제곱 = 편차 ** 2
    # N , 1, 2,
    반지름 = np.sqrt( 제곱.sum((1,2)))
    # N
    반지름평균 = 반지름.mean()
    if(반지름평균 > minR):
        반지름비율 = 반지름 / 반지름평균 # 다른언어에서 변환 타입 주의! 파이썬은 자동으로 실수
        for i in range(반지름비율.size): # iter를 줄이고 싶으면 반만 해도 됨.
            if (1 + errorRate < 반지름비율[i]) or (반지름비율[i] < 1 - errorRate):
                return False
    else:
        return False
    return True

cap = cv2.VideoCapture('video2.mp4')
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
              int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), 1)

lower_red = np.array([0, 50, 50])
upper_red = np.array([20, 255, 255])
lower_blue = np.array([100, 150, 0],np.uint8)
upper_blue = np.array([140, 255, 255],np.uint8)
kernel = np.ones((4, 4), np.uint8)

while True:
    retval, frame_out = cap.read() # 프레임 캡처
    if(retval == None):
        break;
    frame = frame_out[0:frame_size[0] *3 // 4, :]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_red, upper_red)
    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)
    dilation = cv2.dilate(mask, kernel, iterations=1)
    dilation1 = cv2.dilate(mask1, kernel, iterations=1)

    contours, _ = cv2.findContours(
        dilation, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    contours1, _ = cv2.findContours(
        dilation1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        approx = cv2.approxPolyDP(
            cnt, 0.07 * cv2.arcLength(cnt, False), True)
        if len(approx) == 3:
            if(findSameAngle(approx)):
                approx.shape = [1,3,2]
                cv2.polylines(frame, approx, True, (0,0,255), 2, cv2.LINE_4)
                
    for cnt in contours1:
        approx = cv2.approxPolyDP(
            cnt, 0.07 * cv2.arcLength(cnt, False), True)
        if len(approx) == 4:
            if(findSameAngle(approx)):
                approx.shape = [1,4,2]
                cv2.polylines(frame, approx, True, (255,0,0), 5, cv2.LINE_4)
                
    cv2.imshow('frame_out',frame_out)
    key = cv2.waitKey(25)
    if key == 27:
        break

if cap.isOpened():
    cap.release()
cv2.destroyAllWindows()
