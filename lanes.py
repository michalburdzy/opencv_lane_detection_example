import cv2
import numpy as np
import matplotlib.pyplot as plt
from sys import argv

def find_region_of_interest(image):
  height = image.shape[0]
  width = image.shape[1]
  width_margin = 50
  height_margin = 250
  # define area of interest in shape of triangle
  return np.array([
      [(width_margin, height), (width - width_margin, height), (int(width/2), height_margin)]
    ])

def canny(image):
  # reduce image to one color channel
  gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
  # smoothen/blur single contrasting pixels
  blur = cv2.GaussianBlur(gray, (5,5), 0)
  # find edges in the image
  canny = cv2.Canny(blur, 50, 110)

  return canny

def region_of_interest(image):
  polygons = find_region_of_interest(image)
  # create array of zeros of shape identical to image, but black
  mask = np.zeros_like(image)
  # fill in the mask within area of interest (triangle) with white
  cv2.fillPoly(mask, polygons, 255)
  # apply the mask to the image using bitwise AND
  masked_image = cv2.bitwise_and(mask, image)

  return masked_image

def display_lines(image, lines):
  # create black mask image
  line_image = np.zeros_like(image)
  if lines is not None:
    for line in lines:
      x1, y1, x2, y2 = line.reshape(4)
      cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
  return line_image

def average_slope_intercept(image, lines):
  left_fit = []
  right_fit = []
  for line in lines:
    x1, y1, x2, y2 = line.reshape(4)
    parameters = np.polyfit((x1,x2), (y1,y2), 1)
    slope = parameters[0]
    if slope < 0:
      left_fit.append(parameters)
    else:
      right_fit.append(parameters)
  left_fit_avg = np.average(left_fit, axis=0)
  right_fit_avg = np.average(right_fit, axis=0)
  left_line = make_coordinates(image, left_fit_avg)
  right_line = make_coordinates(image, right_fit_avg)

  return np.array([left_line, right_line])

def make_coordinates(image, line_parameters):
  slope, intercept = line_parameters
  y1 = image.shape[0]
  # mark bottom 3/5 of image
  y2 = int(y1*(3/5))
  x1 = int((y1 - intercept) / slope)
  x2 = int((y2 - intercept) / slope)
  return np.array([x1, y1, x2, y2])

def analyze_frame(image):
  contours = canny(image) 
  masked_img = region_of_interest(contours)
  # 2 pixels and 1 degree of precision (in radians). 1 degree = PI / 180``
  lines = cv2.HoughLinesP(masked_img, 1.5, np.pi / 360, 100, np.array([]), minLineLength=40, maxLineGap=5)
  # line_image = display_lines(image, lines)
  avg_lines = average_slope_intercept(image, lines)
  line_image = display_lines(image, avg_lines)

  combined_image = cv2.addWeighted(image, 0.8, line_image, 1, 10)
  return combined_image

def analyze_image():
  image = cv2.imread('images/test_image_0.jpg')
  lane_image_copy = np.copy(image)
  combined_image = analyze_frame(lane_image_copy)
  cv2.imshow('result', combined_image)
  cv2.waitKey(0)

def analyze_video():
  cap = cv2.VideoCapture('videos/test.mp4')
  
  while cap.isOpened():
    _,frame = cap.read()
    combined_image = analyze_frame(frame)
    cv2.imshow('result', combined_image)
    if cv2.waitKey(1) == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()


def main():
  if argv[1] == 'video':
    analyze_video()
  elif argv[1] == 'photo':
    analyze_image()
  else:
    print("Missing required argument: \"video\" or \"photo\"")

main()