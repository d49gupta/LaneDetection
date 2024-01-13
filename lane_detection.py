import cv2
import numpy as np
from Cannygpt import gaussian_kernel
from Cannygpt import canny_edge_detection 

def perform_canny_edge_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0) 
    edges = cv2.Canny(blurred, 50, 150)

    return edges

def detect_lines(image):
    edges = perform_canny_edge_detection(image)

    # Use Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=50, minLineLength=50, maxLineGap=5)

    return lines


def line_length(x1, y1, x2, y2):
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def draw_lines(image, lines, threshold):
    blank = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if line_length(x1, y1, x2, y2) >= threshold:
                cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Draw white lines on blank image only if line lenth is greater than 100
            
            cv2.line(blank, (x1, y1), (x2, y2), (255, 255, 255), 2)

    return image, blank

def lane_detection(frame, height, width):
    # frame = cv2.resize(image, (640, 480), interpolation=cv2.INTER_AREA)
    # frame = perform_canny_edge_detection(image)  

    roi = frame[0:height,0:width] # region of interest, y, x, cutoff bottom portion of image
    
    white_index = []
    mid_point_line = 0

    for index, value in enumerate(roi[180, :]):  #all values of x, 1 column of y, find white lines
        if np.any(value == 255): #has to be white and long enough to be a lane
            white_index.append(index)
    
    if len(white_index) >= 2: #if two or more white lines
        # cv2.circle(img=frame, center=(white_index[0], 150), radius=4, color=(255, 0, 0), thickness=2)  # draw circle at first line
        # cv2.circle(img=frame, center=(white_index[-1], 150), radius=4, color=(255, 0, 0), thickness=2)  # draw circle at end line

        mid_point_line = int((white_index[0] + white_index[-1]) / 2) #midpoint of circles
        print(len(white_index), white_index[0], white_index[-1], mid_point_line)
        cv2.circle(img=frame, center=(mid_point_line, 250), radius=10, color=(255, 0, 0), thickness=3)  # draw circle at midpoint of two circles 

    return frame    
    
if __name__ == "__main__":
    cap = cv2.VideoCapture(r'C:\Users\16134\OneDrive - University of Waterloo\Documents\School\Second Year\2B\MTE 203\Project 2\Images\car_video.mp4')  # Use 0 for webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, channels = frame.shape
        # frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA) #both images must be same size 
        lines = detect_lines(frame)
        image, blank = draw_lines(frame, lines, 210)
        lane = lane_detection(image, height, width)

        combined_frames = cv2.hconcat([blank, lane]) # Display original video and edge detection video
        cv2.imshow('Lane Detection', combined_frames)
 
        # Break the loop when the 'Enter' key (keycode 13) is pressed
        if cv2.waitKey(1) == 13: #waitKey(1) is original video speed, higher you go, slower the video outputs 
            break