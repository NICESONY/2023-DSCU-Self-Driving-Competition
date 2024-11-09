#!/usr/bin/env python
# -*- coding: utf-8 -*- 16
#=============================================
# 본 프로그램은 자이트론에서 제작한 것입니다.
# 상업 라이센스에 의해 제공되므로 무단배포 및 상업적 이용을 금합니다.
# 자이카 C모델 Ubuntu 20.04 + ROS Noetic
#=============================================
#=============================================
# 함께 사용되는 각종 파이썬 패키지들의 import 선언부
#=============================================
import numpy as np
import cv2, time, math ,rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int32MultiArray
from xycar_msgs.msg import xycar_motor
from cv_bridge import CvBridge


#=============================================
# 프로그램에서 사용할 변수, 저장공간 선언부
#=============================================
motor = None  # 모터 노드 변수
Fix_Speed = 23 # 모터 속도 고정 상수값 
new_angle = 0  # 모터 조향각 초기값
new_speed = Fix_Speed  # 모터 속도 초기값
bridge = CvBridge()  # OpenCV 함수를 사용하기 위한 브릿지 
ultra_msg = None  # 초음파 데이터를 담을 변수
image = np.empty(shape=[0])  # 카메라 이미지를 담을 변수
motor_msg = xycar_motor()  # 카메라 토픽 메시지
WIDTH, HEIGHT = 640, 480  # 카메라 이미지 가로x세로 크기
Blue = (0,255,0) # 파란색           #######이거 잘못 된거 아님???#######
Green = (0,255,0) # 녹색       # opencv는 BGR 임
Red = (0,0,255) # 빨간색
Yellow = (0,255,255) # 노란색
stopline_num = 1 # 정지선 발견때마다 1씩 증가
View_Center = WIDTH//2  # 화면의 중앙값 = 카메라 위치

#=============================================
# 차선 인식 프로그램에서 사용할 상수 선언부
#=============================================
#### 이미지의 관심영역 ROI( Region of Interest ) #####@@
## 상수인데도 gobal로 안 불러 와도 되나???####

CAM_FPS = 30  # 카메라 FPS 초당 30장의 사진을 보냄  
WIDTH, HEIGHT = 640, 480  # 카메라 이미지 가로x세로 크기
ROI_START_ROW = 300  # 차선을 찾을 ROI 영역의 시작 Row값
ROI_END_ROW = 380  # 차선을 찾을 ROT 영역의 끝 Row값
ROI_HEIGHT = ROI_END_ROW - ROI_START_ROW  # ROI 영역의 세로 크기  
L_ROW = 40  # 차선의 위치를 찾기 위한 ROI 안에서의 기준 Row값 

#=============================================
# 콜백함수 - USB 전방카메라 토픽을 처리하는 콜백함수.
#=============================================
def usbcam_callback(data):
    global image
    image = bridge.imgmsg_to_cv2(data, "bgr8")




#=============================================
# 콜백함수 - 초음파 토픽을 처리하는 콜백함수.
#=============================================
def ultra_callback(data):
    global ultra_msg, ultra_data
    ultra_msg = data.data




#=============================================
# 모터 토픽을 발행하는 함수.  
#=============================================
def drive(angle, speed):
    motor_msg.angle = angle
    motor_msg.speed = speed
    motor.publish(motor_msg)
    


#=============================================
# 차량을 정차시키는 함수.  
# 입력으로 시간(초)를 받아 그 시간동안 속도=0 토픽을 모터로 보냄
#=============================================
def stop_car(sleep_sec):
    for i in range(sleep_sec*100): 
        drive(angle=new_angle, speed=0)
        time.sleep(0.01)



    
#=============================================
# 초음파 센서를 이용해서 벽까지의 거리를 알아내서
# 벽과 충돌하지 않으며 주행하도록 핸들 조정함.
#=============================================
def sonic_drive():
    global new_angle, new_speed

    # 앞쪽 가까이에 장애물이 있으면 차량 멈춤
    if (0 < ultra_msg[2] < 5):
        new_angle = new_angle
        new_speed = 0
        print("Car Brake, Stop! : ", ultra_msg)

    elif (ultra_msg[1] - ultra_msg[3] > 10):
        new_angle = -60
        new_speed = Fix_Speed
        print("Turn left : ", ultra_msg)
     
    elif (ultra_msg[3] - ultra_msg[1] > 10):
        new_angle = 60
        new_speed = Fix_Speed
        print("Turn right : ", ultra_msg)

    # 위 조건에 해당하지 않는 경우라면 (오른쪽과 왼쪽이 동일한 경우) 똑바로 직진 주행
    else:
        new_angle = 0
        new_speed = Fix_Speed  #### 여기 속도 올리는거 어때?#####
        print("Go Straight : ", ultra_msg)

    # 모터에 주행명령 토픽을 보낸다
    drive(new_angle, new_speed)

#=============================================
# 카메라 이미지를 영상처리하여 
# 정지선이 있는지 체크하고 True 또는 False 값을 반환.
#=============================================

def check_stopline():
    global stopline_num

    # 원본 영상을 화면에 표시
    #cv2.imshow("Original Image", image)
    
    # image(원본이미지)의 특정영역(ROI Area)을 잘라내기
    roi_img = image[250:480, 0:640] 
    cv2.imshow("ROI Image", roi_img)



    ###Hue(색조), Saturation(채도), Value(명도), 3가지 성분의 조합으로 표현#####
    #### Hue(색조) : 색의 종류. 0º~360º의 범위를 갖는다.#####
    #### Saturation(채도) : 색의 선명도, 진함의 정도(가장 진한 상태를 100%로 한다)#####
    #### Value(명도) : 색의 밝기, 밝은 정도(가장 밝은 상태를 100%로 한다)######

    # HSV 포맷으로 변환하고 V채널에 대해 범위를 정해서 흑백이진화 이미지로 변환
    hsv_image = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV) 
    upper_white = np.array([255, 255, 255])
    lower_white = np.array([0, 0, 235])   ###
    binary_img = cv2.inRange(hsv_image, lower_white, upper_white)
    cv2.imshow("Black&White Binary Image", binary_img)  ####



    # 흑백이진화 이미지에서 특정영역을 잘라내서 정지선 체크용 이미지로 만들기
    stopline_check_img = binary_img[120:150, 150:440]  ###
    cv2.imshow("Stopline Check Image", stopline_check_img)  ###


    
    # 흑백이진화 이미지를 칼라이미지로 바꾸고 정지선 체크용 이미지 영역을 녹색사각형으로 표시
    img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img, (150,120),(440,150),Green,3)  ###
    cv2.imshow('Stopline Check', img)
    cv2.waitKey(1)

    
    # 정지선 체크용 이미지에서 흰색 점의 개수 카운트하기
    stopline_count = cv2.countNonZero(stopline_check_img)


    
    # 사각형 안의 흰색 점이 기준치 이상이면 정지선을 발견한 것으로 한다
    if stopline_count > 2700:  ###
        print("Stopline Found...! -", stopline_num)
        stopline_num = stopline_num + 1
        cv2.destroyWindow("ROI Image")
        return True
    
    else:
        return False
        
#=============================================
# 카메라 영상 이미지에서 차선을 찾아 그 위치를 반환하는 코드
#=============================================
def lane_detect():  # hough line code 

    global image
    prev_x_left = 0
    prev_x_right = WIDTH

    img = image.copy() # 이미지처리를 위한 카메라 원본이미지 저장
    display_img = img  # 디버깅을 위한 디스플레이용 이미지 저장
    
    # img(원본이미지)의 특정영역(ROI Area)을 잘라내기
    roi_img = img[ROI_START_ROW:ROI_END_ROW, 0:WIDTH]
    line_draw_img = roi_img.copy()

    #=========================================
    # 원본 칼라이미지를 그레이 회색톤 이미지로 변환하고 
    # 블러링 처리를 통해 노이즈를 제거한 후에 (약간 뿌옇게, 부드럽게)
    # Canny 변환을 통해 외곽선 이미지로 만들기
    #=========================================
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    blur_gray = cv2.GaussianBlur(gray,(5, 5), 0)
    edge_img = cv2.Canny(np.uint8(blur_gray), 60, 75)

    # 잘라낸 이미지에서 HoughLinesP 함수를 사용하여 선분들을 찾음
    all_lines = cv2.HoughLinesP(edge_img, 1, math.pi/180,50,50,20)
    
    if all_lines is None:
        return False, 0, 0

    #=========================================
    # 선분들의 기울기 값을 각각 모두 구한 후에 리스트에 담음. 
    # 기울기의 절대값이 너무 작은 경우 (수평선에 가까운 경우)
    # 해당 선분을 빼고 담음. 
    #=========================================
    slopes = []
    filtered_lines = []

    for line in all_lines:
        x1, y1, x2, y2 = line[0]

        if (x2 == x1):
            slope = 1000.0
        else:
            slope = float(y2-y1) / float(x2-x1)
    
        if 0.2 < abs(slope):
            slopes.append(slope)
            filtered_lines.append(line[0])

    if len(filtered_lines) == 0:
        return False, 0, 0

    #=========================================
    # 왼쪽 차선에 해당하는 선분과 오른쪽 차선에 해당하는 선분을 구분하여 
    # 각각 별도의 리스트에 담음.
    #=========================================
    left_lines = []
    right_lines = []

    for j in range(len(slopes)):
        Line = filtered_lines[j]
        slope = slopes[j]

        x1,y1, x2,y2 = Line

        # 기울기 값이 음수이고 화면의 왼쪽에 있으면 왼쪽 차선으로 분류함
        # 기준이 되는 X좌표값 = (화면중심값 - Margin값)
        Margin = 0
        
        if (slope < 0) and (x2 < WIDTH/2-Margin):
            left_lines.append(Line.tolist())

        # 기울기 값이 양수이고 화면의 오른쪽에 있으면 오른쪽 차선으로 분류함
        # 기준이 되는 X좌표값 = (화면중심값 + Margin값)
        elif (slope > 0) and (x1 > WIDTH/2+Margin):
            right_lines.append(Line.tolist())

    # 디버깅을 위해 차선과 관련된 직선과 선분을 그리기 위한 도화지 준비
    line_draw_img = roi_img.copy()
    
    # 왼쪽 차선에 해당하는 선분은 빨간색으로 표시
    for line in left_lines:
        x1,y1, x2,y2 = line
        cv2.line(line_draw_img, (x1,y1), (x2,y2), Red, 2)

    # 오른쪽 차선에 해당하는 선분은 노란색으로 표시
    for line in right_lines:
        x1,y1, x2,y2 = line
        cv2.line(line_draw_img, (x1,y1), (x2,y2), Yellow, 2)

    #=========================================
    # 왼쪽/오른쪽 차선에 해당하는 선분들의 데이터를 적절히 처리해서 
    # 왼쪽차선의 대표직선과 오른쪽차선의 대표직선을 각각 구함.
    # 기울기와 Y절편값으로 표현되는 아래와 같은 직선의 방적식을 사용함.
    # (직선의 방정식) y = mx + b (m은 기울기, b는 Y절편)
    #=========================================

    # 왼쪽 차선을 표시하는 대표직선을 구함        
    m_left, b_left = 0.0, 0.0
    x_sum, y_sum, m_sum = 0.0, 0.0, 0.0

    # 왼쪽 차선을 표시하는 선분들의 기울기와 양끝점들의 평균값을 찾아 대표직선을 구함
    size = len(left_lines)
    if size != 0:
        for line in left_lines:
            x1, y1, x2, y2 = line
            x_sum += x1 + x2
            y_sum += y1 + y2
            if(x2 != x1):
                m_sum += float(y2-y1)/float(x2-x1)
            else:
                m_sum += 0                
            
        x_avg = x_sum / (size*2)
        y_avg = y_sum / (size*2)
        m_left = m_sum / size
        b_left = y_avg - m_left * x_avg

        if m_left != 0.0:
            #=========================================
            # (직선 #1) y = mx + b 
            # (직선 #2) y = 0
            # 위 두 직선의 교점의 좌표값 (x1, 0)을 구함.           
            x1 = int((0.0 - b_left) / m_left)

            #=========================================
            # (직선 #1) y = mx + b 
            # (직선 #2) y = ROI_HEIGHT
            # 위 두 직선의 교점의 좌표값 (x2, ROI_HEIGHT)을 구함.               
            x2 = int((ROI_HEIGHT - b_left) / m_left)

            # 두 교점, (x1,0)과 (x2, ROI_HEIGHT)를 잇는 선을 그림
            cv2.line(line_draw_img, (x1,0), (x2,ROI_HEIGHT), Blue, 2)

    # 오른쪽 차선을 표시하는 대표직선을 구함      
    m_right, b_right = 0.0, 0.0  
    x_sum, y_sum, m_sum = 0.0, 0.0, 0.0

    # 오른쪽 차선을 표시하는 선분들의 기울기와 양끝점들의 평균값을 찾아 대표직선을 구함
    size = len(right_lines)
    if size != 0:
        for line in right_lines:
            x1, y1, x2, y2 = line
            x_sum += x1 + x2
            y_sum += y1 + y2
            if(x2 != x1):
                m_sum += float(y2-y1)/float(x2-x1)
            else:
                m_sum += 0     
       
        x_avg = x_sum / (size*2)
        y_avg = y_sum / (size*2)
        m_right = m_sum / size
        b_right = y_avg - m_right * x_avg

        if m_right != 0.0:
            #=========================================
            # (직선 #1) y = mx + b 
            # (직선 #2) y = 0
            # 위 두 직선의 교점의 좌표값 (x1, 0)을 구함.           
            x1 = int((0.0 - b_right) / m_right)

            #=========================================
            # (직선 #1) y = mx + b 
            # (직선 #2) y = ROI_HEIGHT
            # 위 두 직선의 교점의 좌표값 (x2, ROI_HEIGHT)을 구함.               
            x2 = int((ROI_HEIGHT - b_right) / m_right)

            # 두 교점, (x1,0)과 (x2, ROI_HEIGHT)를 잇는 선을 그림
            cv2.line(line_draw_img, (x1,0), (x2,ROI_HEIGHT), Blue, 2)

    #=========================================
    # 차선의 위치를 찾기 위한 기준선(수평선)은 아래와 같음.
    #   (직선의 방정식) y = L_ROW 
    # 위에서 구한 2개의 대표직선, 
    #   (직선의 방정식) y = (m_left)x + (b_left)
    #   (직선의 방정식) y = (m_right)x + (b_right)
    # 기준선(수평선)과 대표직선과의 교점인 x_left와 x_right를 찾음.
    #=========================================

    #=========================================        
    # 대표직선의 기울기 값이 0.0이라는 것은 직선을 찾지 못했다는 의미임
    # 이 경우에는 교점 좌표값을 기존 저장해 놨던 값으로 세팅함 
    #=========================================
    if m_left == 0.0:
        x_left = prev_x_left  # 변수에 저장해 놓았던 이전 값을 가져옴

    #=========================================
    # 아래 2개 직선의 교점을 구함
    # (직선의 방정식) y = L_ROW  
    # (직선의 방정식) y = (m_left)x + (b_left)
    #=========================================
    else:
        x_left = int((L_ROW - b_left) / m_left)
                        
    #=========================================
    # 대표직선의 기울기 값이 0.0이라는 것은 직선을 찾지 못했다는 의미임
    # 이 경우에는 교점 좌표값을 기존 저장해 놨던 값으로 세팅함 
    #=========================================
    if m_right == 0.0:
        x_right = prev_x_right  # 변수에 저장해 놓았던 이전 값을 가져옴	
	
    #=========================================
    # 아래 2개 직선의 교점을 구함
    # (직선의 방정식) y = L_ROW  
    # (직선의 방정식) y = (m_right)x + (b_right)
    #=========================================
    else:
        x_right = int((L_ROW - b_right) / m_right)
       
    #=========================================
    # 대표직선의 기울기 값이 0.0이라는 것은 직선을 찾지 못했다는 의미임
    # 이 경우에 반대쪽 차선의 위치 정보를 이용해서 내 위치값을 정함 
    #=========================================
    if m_left == 0.0 and m_right != 0.0:
        x_left = x_right - 380

    if m_left != 0.0 and m_right == 0.0:
        x_right = x_left + 380

    # 이번에 구한 값으로 예전 값을 업데이트 함			
    prev_x_left = x_left
    prev_x_right = x_right
	
    # 왼쪽 차선의 위치와 오른쪽 차선의 위치의 중간 위치를 구함
    x_midpoint = (x_left + x_right) // 2 

    #=========================================
    # 디버깅용 이미지 그리기
    # (1) 수평선 그리기 (직선의 방정식) y = L_ROW 
    # (2) 수평선과 왼쪽 대표직선과의 교점 위치에 작은 녹색 사각형 그리기 
    # (3) 수평선과 오른쪽 대표직선과의 교점 위치에 작은 녹색 사각형 그리기 
    # (4) 왼쪽 교점과 오른쪽 교점의 중점 위치에 작은 파란색 사각형 그리기
    # (5) 화면의 중앙점 위치에 작은 빨간색 사각형 그리기 
    #=========================================
    cv2.line(line_draw_img, (0,L_ROW), (WIDTH,L_ROW), Yellow, 2)
    cv2.rectangle(line_draw_img, (x_left-5,L_ROW-5), (x_left+5,L_ROW+5), Green, 4)
    cv2.rectangle(line_draw_img, (x_right-5,L_ROW-5), (x_right+5,L_ROW+5), Green, 4)
    cv2.rectangle(line_draw_img, (x_midpoint-5,L_ROW-5), (x_midpoint+5,L_ROW+5), Blue, 4)
    cv2.rectangle(line_draw_img, (View_Center-5,L_ROW-5), (View_Center+5,L_ROW+5), Red, 4)

    # 위 이미지를 디버깅용 display_img에 overwrite해서 화면에 디스플레이 함
    display_img[ROI_START_ROW:ROI_END_ROW, 0:WIDTH] = line_draw_img
    cv2.imshow("Lanes positions", display_img)
    cv2.waitKey(1)

    return True, x_left, x_right

#=============================================
# 실질적인 메인 함수 
#=============================================
def start():

    global motor, ultra_msg, image, img_ready 
    global new_angle, new_speed

    SENSOR_DRIVE = 1
    LANE_DRIVE = 2  ### 이거 뭔지 모르겠음 ###
    FINISH = 9
    drive_mode = SENSOR_DRIVE
    
    #=========================================
    # 노드를 생성하고, 구독/발행할 토픽들을 선언합니다.
    #=========================================
    rospy.init_node('Track_Driver')
    motor = rospy.Publisher('xycar_motor', xycar_motor, queue_size=1)
    rospy.Subscriber("xycar_ultrasonic", Int32MultiArray, ultra_callback, queue_size=1)
    rospy.Subscriber("/usb_cam/image_raw/",Image,usbcam_callback, queue_size=1)

    #=========================================
    # 첫번째 토픽이 도착할 때까지 기다립니다.
    #=========================================
    rospy.wait_for_message("/usb_cam/image_raw/", Image)
    print("Camera Ready --------------")
    rospy.wait_for_message("xycar_ultrasonic", Int32MultiArray)
    print("UltraSonic Ready ----------")

    #=========================================
    # 메인 루프 
    #=========================================
    while not rospy.is_shutdown():

        # ======================================
        # 초음파 센서로 주행합니다.
        # 정지선이 보이면 OBJECT_DETECTION 모드로 변경합니다.
        # ======================================
        while drive_mode == SENSOR_DRIVE:
            sonic_drive()
            result = check_stopline()
             
            if (result == True):
                stop_car(1)
                drive_mode = LANE_DRIVE  
                print ("----- Lane driving Start... -----")
                
        # ======================================
        # 차선을 보고 주행합니다.
        # 
        # ======================================
        while drive_mode == LANE_DRIVE:
            found, x_left, x_right = lane_detect()
            if found:
                x_midpoint = (x_left + x_right) // 2 
                new_angle = (x_midpoint - View_Center) // 3
                drive(new_angle, new_speed)
            else:
                drive(new_angle, new_speed)
                
            result = check_stopline()
            if (result == True):
                drive_mode = FINISH  

        # ======================================
        # 주행을 끝냅니다. 
        # ======================================
        if drive_mode == FINISH:
            # 차량을 정지시키고 모든 작업을 끝냅니다.
            stop_car(1)  
            time.sleep(2)
            print ("----- Bye~! -----")
            return            

#=============================================
# 메인함수 호툴
# start() 함수가 실질적인 메인함수임. 
#=============================================
if __name__ == '__main__':
    start()

## roslaunch track_drive track_drive.launch

