import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils          # mediapipe 繪圖方法
mp_drawing_styles = mp.solutions.drawing_styles  # mediapipe 繪圖樣式
mp_pose = mp.solutions.pose                      # mediapipe 姿勢偵測

# 寫入影像並儲存影片
# cap = cv2.VideoCapture(0)         # 讀取攝影鏡頭
cap = cv2.VideoCapture('D:\exp｜mediapipe\ignore\IMG_3702.mov') # 讀取電腦中的影片
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    # 取得影像寬度
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 取得影像高度
fps = int(cap.get(cv2.CAP_PROP_FPS))              # 取得影像幀率
fourcc = cv2.VideoWriter_fourcc(*'MJPG')          # 設定影片的格式為 MJPG
resize_wid = round(width/1.5)                     # 更改影像寬度
resize_hei = round(height/1.5)                    # 更改影像高度
out = cv2.VideoWriter('D:\exp｜mediapipe\ignore\pose_output.mp4', fourcc, fps, (resize_wid,  resize_hei))  # 產生空的影片

# 啟用姿勢偵測
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, img = cap.read()
        if not ret:
            print("Cannot receive frame")
            break
        img = cv2.resize(img,(resize_wid,resize_hei)) # 縮小尺寸，加快演算速度
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # 將 BGR 轉換成 RGB
        results = pose.process(img2)                  # 取得姿勢偵測結果
        # 根據姿勢偵測結果，標記身體節點和骨架
        mp_drawing.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        out.write(img)       # 將取得的每一幀圖像寫入空的影片
        cv2.imshow('pose_output', img)
        if cv2.waitKey(5) == ord('q'):
            break     # 按下 q 鍵停止
cap.release()
# out.release()      # 釋放資源
cv2.destroyAllWindows()