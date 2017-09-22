import cv2
import numpy as np
#import matplotlib.py


def filter_region(image, vertices):
    """
    Create the mask using the vertices and apply it to the input image
    """
    mask = np.zeros_like(image)
    if len(mask.shape)==2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,)*mask.shape[2]) # in case, the input image has a channel dimension        
    return cv2.bitwise_and(image, mask)

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


img = cv2.imread('solidWhiteRight.jpg')

lower_white = np.uint8([0,200,0])
upper_white = np.uint8([255,255,255])

white = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

white_mask = cv2.inRange(white, lower_white,upper_white)
masked = cv2.bitwise_and(img, img, mask = white_mask)   #Filter out all colors except for white

gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(13,13),0)
edges = cv2.Canny(blur, 50,150)
edges = select_region(edges)

lines = cv2.HoughLinesP(edges,rho = 1,theta = np.pi/180,threshold = 20,minLineLength = 20,maxLineGap = 300)

for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
    
cv2.imshow('res',img)









if cv2.waitKey(0) & 0xFF == ord('q'):
	cv2.destroyAllWindows()
