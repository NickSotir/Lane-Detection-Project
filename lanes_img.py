import cv2
import numpy as np
import math
from abc import ABCMeta


def filter_white_yellow(image):

    hsl_img = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    lower_white = np.uint8([[0,200,0]])
    upper_white = np.uint8([[255,255,255]])
    white_color = cv2.inRange(hsl_img, lower_white, upper_white)

    lower_yellow = np.uint8([[10,0,100]])
    upper_yellow = np.uint8([[40, 255, 255]])
    yellow_color = cv2.inRange(hsl_img, lower_yellow, upper_yellow)

    white_and_yellow = cv2.bitwise_or(white_color, yellow_color)    #Combine the two masks in order to detect both white and yellow
    masked = cv2.bitwise_and(image, image, mask = white_and_yellow)
    return masked
def filter_region(image, vertices):
    """
    Create the mask using the vertices and apply it to the input image
    """
    mask = np.zeros_like(image)
    if len(mask.shape)==2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # in case, the input image has a channel dimension        
    return cv2.bitwise_and(image,mask)
def select_region(image):
    """
    It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
    """
    # first, define the polygon by vertices
    rows, cols = image.shape[:2]
    bottom_left  = [cols*0.1, rows*0.95]
    top_left     = [cols*0.4, rows*0.6]
    bottom_right = [cols*0.9, rows*0.95]
    top_right    = [cols*0.6, rows*0.6] 
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return filter_region(image, vertices)   
def Gaussian_Blur(image, kernel_size):

    ##### Gaussian Blurring improves accuracy when it comes to edge detection #####
    return cv2.GaussianBlur(image,(kernel_size,kernel_size),0)
def Hough_line(image, minLength, maxGap, thresh):
    
    return cv2.HoughLinesP(image, rho = 1, theta = np.pi/180, threshold = thresh, minLineLength = minLength, maxLineGap = maxGap)
def avg_lines(lines): #Compute the weighted average of the lines found
    right_lane = []
    right_weights = []
    left_lane = []
    left_weights = []

    for line in lines:
        for x1,y1,x2,y2 in line:
            if x2==x1:
                continue
            slope = (y2-y1)/(x2-x1)
            intercept = y1 - slope*x1
            length = np.sqrt((y2-y1)**2 +(x2-x1)**2)

        if slope < 0:                   #If slope is negative then it is a left lane
            left_lane.append((slope,intercept))
            left_weights.append(length)
        else:                           #Else it is the right lane                           
            right_lane.append((slope, intercept))
            right_weights.append(length)

    if len(left_weights)>0:  
        avg_left_lane = np.dot(left_weights, left_lane)/np.sum(left_weights) 
    else:
        None
    if len(right_weights)>0:
        avg_right_lane = np.dot(right_weights,right_lane)/np.sum(right_weights) 
    else:
        None

    return avg_right_lane, avg_left_lane
def Canny_Edge(image, min_thresh, max_thresh):

    return cv2.Canny(image, min_thresh, max_thresh)
def compute_line_points(y1, y2, line):
    if line is None:
        return None

    slope, intercept = line
    """ Line Equation: y = mx+b 
        Knowing y, m and b we can easily compute x from the equation x = (y-b)/m 
        Notice that all points must be integers otherwise the cv2.line will throw an exception
    """
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    y1 = int(y1)
    y2 = int(y2)

    return ((x1,y1), (x2,y2)) #Return a tuple with the points for the lines
img = cv2.imread('solidWhiteRight.jpg')
masked = filter_white_yellow(img)
gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
blur = Gaussian_Blur(gray, 13)                                 
edge = Canny_Edge(blur, 50,150)
edges = select_region(edge)

lines = Hough_line(edges, 20,300,20)
avg_lines(lines)

for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),2)

    
cv2.imshow('res',img)

if cv2.waitKey(0) & 0xFF == ord('q'):
	cv2.destroyAllWindows()
