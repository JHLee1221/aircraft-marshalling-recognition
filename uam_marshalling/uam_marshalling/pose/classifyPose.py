from pose.calculateAngle import calculateAngle
import mediapipe as mp
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO
from pose.mpose import mp_pose

#Initalizing YOLOV8 in video
#model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True) # /home/jhlee98/yolov5/", 'custom', path='/home/jhlee98/yolov5/yolov5m.pt', source = 'local

model = YOLO(model="../model/yolov8s-msh.pt")

#Initial settings
color = (0, 0, 255)

class classifyPose(object):
    
    def __init__(self):
        pass
    
    ############# Display landmarks in video #############
    def pose_landmarks(self, landmarks, output_image, display = False):
        
        global replica, results, xy, result, cls
        replica = output_image # Copy image frame
        results = model(replica) # YOLOV8 model in frame
        boxes = results[0].boxes # YOLOV8 convert to tensor
        cls = boxes.cls # YOLOV8's classes
        xy = boxes.xyxy     
        #result = cls
        # for result in results:
        #     boxes = result.boxes  # Boxes object for bbox outputs
        #     cls = boxes.cls
        #     xy = boxes.xyxy
        # Always running YOLOv8
        try:
            cls
            xy
        except IndexError:
            pass
        
        # Set left elbow angle
        self.left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                              landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])

        # Set left shoulder angle
        self.left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                                 landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])

        # Set left knee angle
        self.left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                             landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
        
        # Set right elbow angle
        self.right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                               landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])

        # Set right shoulder angle
        self.right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                                  landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])

        # Set right knee angle
        self.right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                              landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                              landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
        
        # Show display or return frame
        if display:
            plt.figure(figsize=[10,10])
            plt.imshow(replica[:,:,::-1]);plt.title("Output Image");plt.axis('off');

        else:
            return replica
    
    ############# The angle of hover mashall pose and hover pose detection #############
    def hover_angle(self, angle = False):
        if self.left_knee_angle > 165 and self.left_knee_angle < 195 and self.right_knee_angle > 165 and self.right_knee_angle < 195:
            if self.left_shoulder_angle > 75 and self.left_shoulder_angle < 105 and self.right_shoulder_angle > 75 and self.right_shoulder_angle < 105:
                if self.left_elbow_angle > 165 and self.left_elbow_angle < 195 and self.right_elbow_angle > 165 and self.right_elbow_angle < 195:
                    cv2.putText(replica, 'HOVER', (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                    return angle
    
    def hover_detect(self, detect = False):
        if cls.cpu().detach().numpy().tolist() == [0]:
            cv2.rectangle(replica, (int(xy[0][0]), int(xy[0][1])), (int(xy[0][2]), int(xy[0][3])),(0, 255, 0), 2)
            cv2.putText(replica, 'HOVER', (int(xy[0][0]), int(xy[0][1])), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2, cv2.LINE_AA)
            return detect
    
    ############ The angle of land mashall pose and land pose detection #############
    def land_angle(self, angle = False):
        if self.left_knee_angle > 165 and self.left_knee_angle < 195 and self.right_knee_angle > 165 and self.right_knee_angle < 195:
            if self.left_shoulder_angle >0 and self.left_shoulder_angle < 15 and self.right_shoulder_angle > 0 and self.right_shoulder_angle < 15:
                if self.left_elbow_angle > 70 and self.left_elbow_angle < 350 and self.right_elbow_angle > 70 and self.right_elbow_angle < 350:
                    cv2.putText(replica, 'LAND', (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                    return angle
                
    def land_detect(self, detect = False):
        if cls.cpu().detach().numpy().tolist() == [1]:
            cv2.rectangle(replica, (int(xy[0][0]), int(xy[0][1])), (int(xy[0][2]), int(xy[0][3])),(0, 125, 125), 2)
            cv2.putText(replica, 'LAND', (int(xy[0][0]), int(xy[0][1])), cv2.FONT_HERSHEY_PLAIN, 2, (0, 125, 125), 2)
            return detect
                            
    ############# The angle of move downward mashall pose and move downward pose detection #############
    def mv_down_angle(self, angle = True):
        if self.left_knee_angle > 165 and self.left_knee_angle < 195 and self.right_knee_angle > 165 and self.right_knee_angle < 195:
            if self.left_shoulder_angle > 5 and self.left_shoulder_angle < 75 and self.right_shoulder_angle > 5 and self.right_shoulder_angle < 75:
                if self.left_elbow_angle > 165 and self.left_elbow_angle < 195 and self.right_elbow_angle > 165 and self.right_elbow_angle < 195:
                    cv2.putText(replica, 'MOVE DOWNWARD', (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                    return angle

    def mv_down_detect(self, detect = False):
        if cls.cpu().detach().numpy().tolist() == [4]:
            cv2.circle(replica, (int((int(xy[0][0]) + int(xy[0][2]))/2), int((int(xy[0][1]) + int(xy[0][3]))/2)), 5, (125, 125, 0), -1)
            cv2.rectangle(replica, (int(xy[0][0]), int(xy[0][1])), (int(xy[0][2]), int(xy[0][3])),(125, 125, 0), 2)
            #cv2.arrowedLine(replica, (640, 360),  (int((int(xy[0][0]) + int(xy[0][2]))/2), int((int(xy[0][1]) + int(xy[0][3]))/2)), (0, 255, 0), thickness=2)
            cv2.putText(replica, 'MOVE DOWNWARD', (int(xy[0][0]), int(xy[0][1])), cv2.FONT_HERSHEY_PLAIN, 2, (125, 125, 0), 2)
            return detect
                            
    ############# The angle of move left mashall pose and move left pose detection #############
    def mv_left_angle(self, angle = False):
        if self.left_knee_angle > 165 and self.left_knee_angle < 195 and self.right_knee_angle > 165 and self.right_knee_angle < 195:
            if self.left_shoulder_angle > 0 and self.left_shoulder_angle < 75 and self.right_shoulder_angle > 75 and self.right_shoulder_angle < 105:
                if self.right_elbow_angle > 165 and self.right_elbow_angle < 195:
                    cv2.putText(replica, 'MOVE LEFT', (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                    return angle
                
    def mv_left_detect(self, detect = False):
        if cls.cpu().detach().numpy().tolist() == [2]:
            cv2.rectangle(replica, (int(xy[0][0]), int(xy[0][1])), (int(xy[0][2]), int(xy[0][3])),(100, 200, 100), 2)
            cv2.putText(replica, 'MOVE LEFT', (int(xy[0][0]), int(xy[0][1])), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 100), 2)
            return detect
                        
    ############# The angle of move right mashall pose and move right pose detection #############
    def mv_right_angle(self, angle = False):
        if self.left_knee_angle > 165 and self.left_knee_angle < 195 and self.right_knee_angle > 165 and self.right_knee_angle < 195:
            if self.left_shoulder_angle > 75 and self.left_shoulder_angle < 105 and self.right_shoulder_angle > 0 and self.right_shoulder_angle < 75:
                if self.left_elbow_angle > 165 and self.left_elbow_angle < 195:
                    cv2.putText(replica, 'MOVE RIGHT', (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                    return angle
                        
    def mv_right_detect(self, detect = False):
        if cls.cpu().detach().numpy().tolist() == [3]:
            cv2.rectangle(replica, (int(xy[0][0]), int(xy[0][1])), (int(xy[0][2]), int(xy[0][3])),(125, 100, 100), 2)
            cv2.putText(replica, 'MOVE RIGHT', (int(xy[0][0]), int(xy[0][1])), cv2.FONT_HERSHEY_PLAIN, 2, (85, 100, 150), 2)
            return detect
    
    ############# The angle of move upward mashall pose and move upward pose detection #############        
    def mv_up_angle(self, angle = False):
        if self.left_knee_angle > 165 and self.left_knee_angle < 195 and self.right_knee_angle > 165 and self.right_knee_angle < 195:
            if self.left_shoulder_angle > 115 and self.left_shoulder_angle < 185 and self.right_shoulder_angle > 115 and self.right_shoulder_angle < 185:
                if self.left_elbow_angle > 165 and self.left_elbow_angle < 195 and self.right_elbow_angle > 165 and self.right_elbow_angle < 195:
                    cv2.putText(replica, 'MOVE UPWARD', (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                    return angle

    def mv_up_detect(self, detect = False):
        if cls.cpu().detach().numpy().tolist() == [5]:      
            cv2.rectangle(replica, (int(xy[0][0]), int(xy[0][1])), (int(xy[0][2]), int(xy[0][3])),(0, 0, 255), 2)
            cv2.putText(replica, 'MOVE UPWARD', (int(xy[0][0]), int(xy[0][1])), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            return detect

    def arm_angle(self, angle = False):
        if self.left_knee_angle > 165 and self.left_knee_angle < 195 and self.right_knee_angle > 165 and self.right_knee_angle < 195:
            if self.left_shoulder_angle > 5 and self.left_shoulder_angle < 75 and self.right_shoulder_angle > 75 and self.right_shoulder_angle < 105:
                if self.left_elbow_angle > 165 and self.left_elbow_angle < 195 and self.right_elbow_angle > 255 and self.right_elbow_angle < 285:
                    cv2.putText(replica, "ENGAGE ROTOR", (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                    return angle

    def arm_detect(self, detect = False):
        if cls.cpu().detach().numpy().any().tolist() == [8]:    
            cv2.rectangle(replica, (int(xy[0][0]), int(xy[0][1])), (int(xy[0][2]), int(xy[0][3])),(255, 175, 0), 2)
            cv2.putText(replica, 'ENGAGE ROTOR', (int(xy[0][0]), int(xy[0][1])), cv2.FONT_HERSHEY_PLAIN, 2, (255, 175, 0), 2)
            return detect

    def ahead_angle(self, angle = False):
        if self.left_knee_angle > 165 and self.left_knee_angle < 195 and self.right_knee_angle > 165 and self.right_knee_angle < 195:
            if self.left_shoulder_angle > 75 and self.left_shoulder_angle < 105 and self.right_shoulder_angle > 75 and self.right_shoulder_angle < 105:
                if self.left_elbow_angle > 15 and self.left_elbow_angle < 155 and self.right_elbow_angle > 205 and self.right_elbow_angle < 285: #self.left_elbow_angle > 15 and self.left_elbow_angle < 155 and 
                    cv2.putText(replica, "STRAIGHT AHEAD", (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                    return angle

    def ahead_detect(self, detect = False):
        if cls.cpu().detach().numpy().tolist() == [6]:        
            cv2.rectangle(replica, (int(xy[0][0]), int(xy[0][1])), (int(xy[0][2]), int(xy[0][3])),(25, 185, 100), 2)
            cv2.putText(replica, 'STRAIGHT AHEAD', (int(xy[0][0]), int(xy[0][1])), cv2.FONT_HERSHEY_PLAIN, 2, (25, 185, 100), 2)
            
            return detect
            
    def back_angle(self, angle = False):
        if self.left_knee_angle > 165 and self.left_knee_angle < 195 and self.right_knee_angle > 165 and self.right_knee_angle < 195:
            if self.left_shoulder_angle > 30 and self.left_shoulder_angle < 60 and self.right_shoulder_angle > 30 and self.right_shoulder_angle < 60:
                if self.left_elbow_angle > 15 and self.left_elbow_angle < 315 and self.right_elbow_angle > 15 and self.right_elbow_angle < 315: #
                    cv2.putText(replica, "MOVE BACK", (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
                    return angle

    def back_detect(self, detect = False):
        if cls.cpu().detach().numpy().any().tolist() == [7]:     
            cv2.rectangle(replica, (int(xy[0][0]), int(xy[0][1])), (int(xy[0][2]), int(xy[0][3])),(200, 205, 0), 2)
            cv2.putText(replica, 'MOVE BACK', (int(xy[0][0]), int(xy[0][1])), cv2.FONT_HERSHEY_PLAIN, 2, (200, 205, 0), 2)
            return detect
