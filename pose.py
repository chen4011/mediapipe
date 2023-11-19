import cv2
import mediapipe as mp
import csv
mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
mp_pose = mp.solutions.pose                      # mediapipe 姿勢偵測

# 寫入影像並儲存影片
# cap = cv2.VideoCapture(0)         # 讀取攝影鏡頭
cap = cv2.VideoCapture('D:\exp_mediapipe\ignore\IMG_3917.MOV') # 讀取電腦中的影片
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    # 取得影像寬度
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 取得影像高度
fps = int(cap.get(cv2.CAP_PROP_FPS))              # 取得影像幀率
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # 影片總共有幾幀
fourcc = cv2.VideoWriter_fourcc(*'MJPG')          # 設定影片的格式為 MJPG
resize_wid = round(width/1.5)                     # 更改影像寬度
resize_hei = round(height/1.5)                    # 更改影像高度
out = cv2.VideoWriter('D:\exp_mediapipe\ignore\pose_output.mp4', fourcc, fps, (resize_wid,  resize_hei))  # 產生空的影片
print(frame_count)

# 建立要蒐集的座標儲存格
Pos = [[[0 for m in range(3)] for n in range(frame_count)] for o in range(2)]
dis = [[0 for m in range(3)] for n in range(frame_count)]
# print(Pos)
# 要計算的座標點
pt1 = 15
pt2 = 16

# 啟用姿勢偵測
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    # print('+')

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    with open('pose_lmk.txt', 'w') as file:               # 建立 .txt
        while True:
            ret, img = cap.read()
            if not ret:
                print("Cannot receive frame")
                break
            img = cv2.resize(img,(resize_wid,resize_hei)) # 縮小尺寸，加快演算速度
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
            results = pose.process(imgRGB)                # 取得姿勢偵測結果
            frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) # 從 0 開始的被截取或解碼的幀的索引值

            print(frame)
            
            # 根據姿勢偵測結果，標記身體節點和骨架
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    img,                                      # 畫的圖片
                    results.pose_landmarks,                   # 畫的點的landmark
                    mp_pose.POSE_CONNECTIONS,                 # 將畫的點連在一起
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()) # 點or線的樣式
                
                # print('*')

                # 取得兩點座標
                # for i, lm in enumerate(results.pose_landmarks.landmark):
                #     if i == 15 or i == 16:
                #         xPos = int(lm.x * resize_wid)
                #         yPos = int(lm.y * resize_hei)
                #         print(i, xPos, yPos)
                for i, lm in enumerate(results.pose_landmarks.landmark):
                    if i == pt1:
                        xPos = int(lm.x * resize_wid)
                        yPos = int(lm.y * resize_hei)
                        Pos[0][frame-1] = [i, xPos, yPos]
                    elif i == pt2:
                        xPos = int(lm.x * resize_wid)
                        yPos = int(lm.y * resize_hei)
                        Pos[1][frame-1] = [i, xPos, yPos]
                    
                    # 計算兩點距離
                    x1 = float(Pos[1][frame-1][1])
                    y1 = float(Pos[1][frame-1][2])
                    x0 = float(Pos[0][frame-1][1])
                    y0 = float(Pos[0][frame-1][2])

                    dis[frame-1][0] = difx = x1 - x0
                    dis[frame-1][1] = dify = y1 - y0
                    dis[frame-1][2] = (difx ** 2 + dify ** 2) ** 0.5

                # print(".")

            out.write(img)       # 將取得的每一幀圖像寫入空的影片
            cv2.imshow('pose_output', img)
            if cv2.waitKey(5) == ord('q'):
                break     # 按下 q 鍵停止
        print(Pos)
        print(dis)
        file.write(f'{frame_count}\n')
        file.write(f'Position : {Pos}\n')  # Write data to the file
        # file.write('\n')
        file.write(f'distance : {dis}\n')  # Write data to the file
        # file.write('\n')
cap.release()
# out.release()      # 釋放資源
cv2.destroyAllWindows()