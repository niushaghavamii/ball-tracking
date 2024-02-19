from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# define the lower and upper boundaries of the "green"
# ball in the HSV color space, then initialize the
# list of tracked points
greenLower = (20, 100, 100)
greenUpper = (30, 255, 255)
pts = deque(maxlen=args["buffer"])

# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	vs = VideoStream(src=0).start()

# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])
frame = vs.read()

#constants define
fragments_number=16
frame_height = 720
frame_width = 1080


#random numbers to look cool
print(frame_height,frame_width)
print(frame.shape[0],frame.shape[-1])


# allow the camera or video file to warm up 
#time.sleep(2.0)
from cvzone.HandTrackingModule import HandDetector
import random

detector = HandDetector(detectionCon = 0.5, maxHands=2)

# timer = 0
# stateResult = False
# startGame = False
# scors = [0,0]

# keep looping
while True:
	
	# grab the current frame
	frame = vs.read()
	# handle the frame from VideoCapture or VideoStream
	frame = frame[-1] if args.get("video", False) else frame

	# if we are viewing a video and we did not grab a frame,
	# then we have reached the end of the video
	if frame is None:
		break

	# resize the frame, blur it, and convert it to the HSV
	# color space
	
	#resizing everything
	frame =cv2.resize(frame, (frame_width, frame_height))
	cv2.namedWindow("Frame")
	cv2.resizeWindow("Frame", frame_width, frame_height)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	#frame = imutils.resize(frame, width=640)

	hands,img = detector.findHands(frame)
	

	# process hand detection results
	for hand in hands:
		lmList = hand["lmList"]
		
		# Check if all fingers are open or closed
		fingers = detector.fingersUp(hand)
		if fingers is not None:  # Ensure fingersUp() is called only when hand is detected
			if all(fingers):
				cv2.putText(frame, "Open", (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
			else:
				cv2.putText(frame, "Closed", (200, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
			

	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)


	

	# construct a mask for the color "green", then perform
	# a series of dilations and erosions to remove any small
	# blobs left in the mask
	mask = cv2.inRange(hsv, greenLower, greenUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	# find contours in the mask and initialize the current
	# (x, y) center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None

	

	# only proceed if at least one contour was found
	if len(cnts) > 0:
		# find the largest contour in the mask, then use
		# it to compute the minimum enclosing circle and
		# centroid
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		
		# only proceed if the radius meets a minimum size
		if radius > 10:
			# draw the circle and centroid on the frame,
			# then update the list of tracked points
			cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)

	# update the points queue
	pts.appendleft(center)

	# loop over the set of tracked points
	for i in range(1, len(pts)):
		# if either of the tracked points are None, ignore
		# them
		if pts[i - 1] is None or pts[i] is None or pts[-1] is None or pts[-2] is None:
			continue
		#print(pts[-1],pts[-2],"********")
		try:
			if pts[-1][0]==pts[-2][0] and pts[-1][1]==pts[-2][1]:
				continue
		except:
			pass
		# otherwise, compute the thickness of the line and
		# draw the connecting lines
		thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
		cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
		print(pts[i])
		#time.sleep(0.)
  
		#deciding if object is in which area
		print(int((pts[i][0])/(frame_width/fragments_number))+1,'x')
		print(int(pts[i][1]/(frame_height/fragments_number))+1,'y')
	
 
 
 
 
	#DRAWING GRIDS
	height, width = frame_height,frame_width
	fragment_width = frame_width // fragments_number
	fragment_height = frame_height // fragments_number

	# Draw horizontal lines to divide the window into fragments
	for r in range(1, fragments_number):
		y = r * fragment_height
		cv2.line(frame, (0, y), (frame_width, y), (255, 0, 0), 1)

	# Draw vertical lines to divide the window into fragments
	for c in range(1, fragments_number):
		x = c * fragment_width
		cv2.line(frame, (x, 0), (x, frame_height), (255, 0, 0), 1)
  
  
  
  
	# show the frame to our screen
	cv2.namedWindow("Frame")
	# save a frame to an image file
	#cv2.imwrite("frame.png", frame)

	
	#debug numbers to look even cooler
	#x, y, w, h = cv2.getWindowImageRect("Frame")
	#print(f"Window size: {w}x{h}")
	#print(f"Frame size: {frame_width}x{frame_height}")
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key is pressed, stop the loop
	if key == ord("q"):
		# startGame = True
        # intialTime = time.time()
        # stateResult = False
		break

# if we are not using a video file, stop the camera video stream
if not args.get("video", False):
	vs.stop()

# otherwise, release the camera
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()
