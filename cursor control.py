
#IMPORTING LIBRARIES

#for determining position of facial landmarks on face we import imutils and dlib
from imutils import face_utils
import dlib
#importing cv2(opencv) for computer vision
import cv2
#importing pyautogui for integrating code with mouse 
import pyautogui as pag
#importing time for calibrating code at particular time
import time
#importing numpy for matrix operations
import numpy as np
#importing matplotlib for ploting graph of data stored during callibration
from matplotlib import pyplot as plt
#importing scipy for smoothning graphs obtained after calibration
from scipy.ndimage import gaussian_filter

# dimensions of the screen are 1920x1080.
#formula to find distance between two points is defined
def dst(point1,point2):
	distance = np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
	return distance 

# The function is used for calculation of EAR of eye
def EAR(point1,point2,point3,point4,point5,point6):
	ear = (dst(point2,point6) + dst(point3,point5))/(2*dst(point1,point4))*1.0
	return ear

#The function is used for calculating the angle between the line made by the nose tip and centre of reference circle and the horizontal line passing through the centre of the reference circle	
def angle(point1):
	slope12 = (point1[1] - 250)/(point1[0] - 250)*1.0
	angle = np.arctan(slope12)
	return angle

# The function is used for calculation of MAR for the mouth
def MAR(point1,point2,point3,point4,point5,point6,point7,point8):
	mar=(dst(point2,point8) + dst(point3,point7) + dst(point4,point6))/(2*dst(point1,point5))*1.0
	return mar

#START OF MAIN PROGRAM PART 1:

#if .dat which you have installed is not in current directory then first specify add of that directory and then name of .dat file
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector() # Returns a default face detector object
predictor = dlib.shape_predictor(p) # Outputs a set of location points that define a pose of the object. (Here, pose of the human face)
cap = cv2.VideoCapture(0)
initial=time.time()
final1=initial+1
final2=final1+5
final3=final2+5
final4=final3+1
final5=final4+5
final6=final5+1
final7=final6+5
#creating empty arrays/lists
opened=np.array([])
left=np.array([])
right=np.array([])
mouth=np.array([])
areal=np.array([])
arear=np.array([])
t0=np.array([])
t1=np.array([])
t2=np.array([])
t3=np.array([])
t4=np.array([])
t5=np.array([])
#font used for putting text on the screen(will be used in sometime)
font = cv2.FONT_HERSHEY_SIMPLEX
#starting & ending point of moth is stored in mstart & mend
(mstart, mend)=face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
#since in code we flip frame but we don't flip landmarks so here rstart,rend stores starting & ending point of left eye same for right eye
(rstart,rend)=face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(lstart,lend)=face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
#calibration code will run for 26 sec
while ((time.time()-initial)<=26):
	# Getting out image by webcam 
	ret,image = cap.read() 
	blackimage = np.zeros((480,640,3),dtype = np.uint8)
	image=cv2.flip(image,1)
	# Converting the image to gray scale image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#using contrast limited adaptive histogram equalization for clarifying image and making it more beautiful
	clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
	gray = clahe.apply(gray)
	# Get faces into webcam's image
	rects = detector(gray, 0)
	#if (len(rects))>1:
	#	rects=rects[0]
	#elif (len(rects))==1:
	#	rects=rects
	
	# For each detected face, find the landmark.
	for (i, rect) in enumerate(rects):
		# Make the prediction and transfom it to numpy array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		actual_t=time.time()-initial
		l=EAR(shape[36],shape[37],shape[38],shape[39],shape[40],shape[41])
		r=EAR(shape[42],shape[43],shape[44],shape[45],shape[46],shape[47])
		sub=(l-r)*100
		leftEye = shape[lstart:lend]
		rightEye = shape[rstart:rend]
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(image,[leftEyeHull],-1,(0,255,0),1)
		cv2.drawContours(image,[rightEyeHull],-1,(0,255,0),1)
		larea = cv2.contourArea(leftEyeHull)
		rarea = cv2.contourArea(rightEyeHull)
		mouth_shape = shape[mstart:mend]
		mouth_hull=cv2.convexHull(mouth_shape)
		cv2.drawContours(image,[mouth_hull],-1,(0,255,0),1)
		marea= cv2.contourArea(mouth_hull)	
		sub_area=larea-rarea
		#here sub_area is devided by marea so that code won't be distance dependent
		sub1=sub_area/marea
		x,y,width,height=cv2.boundingRect(leftEyeHull)
		roi1=image[y-20:y+height+20,x-20:x+width+20]
		cv2.imshow('left eye',roi1)
		eyer_hull=cv2.convexHull(rightEyeHull)
		x1,y1,width1,height1=cv2.boundingRect(eyer_hull)
		roi2=image[y1-20:y1+height1+20,x1-20:x1+width1+20]
		cv2.imshow('right eye',roi2)
		
		
		if (final1<time.time()) and (time.time()<final2):
			cv2.putText(blackimage,'Keep Both Eyes Open',(0,100), font, 1,(255,255,255),2,cv2.LINE_AA)
			opened=np.append(opened,[sub])
			t0=np.append(t0,[actual_t])
		
		elif (final2<time.time()) and (time.time()<final3):
			cv2.putText(blackimage,'Close Left Eye',(0,100), font, 1,(255,255,255),2,cv2.LINE_AA)
			left=np.append(left,[sub])
			areal=np.append(areal,[sub1])
			t1=np.append(t1,[actual_t])
										
		elif (final4<time.time()) and (time.time()<final5):
			cv2.putText(blackimage,'Close Right Eye',(0,100), font, 1,(255,255,255),2,cv2.LINE_AA)
			right=np.append(right,[sub])
			arear=np.append(arear,[sub1])
			t2=np.append(t2,[actual_t])

		elif (final6<time.time()) and (time.time()<final7):
			cv2.putText(blackimage,'Open Your Mouth',(0,100), font, 1,(255,255,255),2,cv2.LINE_AA)
			mouth=np.append(mouth,[marea])
			t3=np.append(t3,[actual_t])
	
		else:
			pass

		for (x,y) in shape: # prints facial landmarks on the face
			cv2.circle(image,(x,y),2,(0,255,0),-1)
		res = np.vstack((image,blackimage))
		cv2.imshow('Calibration',res) # Display of image as well as the prompt window
	if cv2.waitKey(5) & 0xFF==ord('h'):
		break

#plotting graphs
#here gaussian filter is used because it gives better results
plt.subplot(3,2,1)
smooth_1=gaussian_filter(left,sigma=5)
plt.title('for ear of left')
plt.plot(t1,smooth_1)
plt.subplot(3,2,2)
smooth_2=gaussian_filter(right,sigma=5)
plt.title('for ear of right')
plt.plot(t2,smooth_2)
plt.subplot(3,2,3)
smooth_3=gaussian_filter(mouth,sigma=5)
plt.title('for area of mouth')
plt.plot(t3,smooth_3)
plt.subplot(3,2,4)
smooth_0=gaussian_filter(opened,sigma=5)
plt.title('both eyes open')
plt.plot(t0,smooth_0)
plt.subplot(3,2,5)
smooth_4=gaussian_filter(areal,sigma=5)
plt.title('for area of left')
plt.plot(t1,smooth_4)
plt.subplot(3,2,6)
smooth_5=gaussian_filter(arear,sigma=5)
plt.title('for area of right')
plt.plot(t2,smooth_5)


plt.show()
cap.release()
cv2.destroyAllWindows()

#START OF MAIN PROGRAM PART 2:
cap=cv2.VideoCapture(0)
sortleft=np.sort(left)#sorting values in list
stdleft=np.std(sortleft)
print(np.std(sortleft))
sortright=np.sort(right)
stdright=np.std(sortleft)
print(np.std(sortright))
sortleftarea=np.sort(areal)
sortrightarea=np.sort(arear)
medianleft=np.median(sortleft)#finding median from sorted values
medianright=np.median(sortright)
medianleftarea=np.median(sortleftarea)
medianrightarea=np.median(sortrightarea)
sortmouth=np.sort(mouth)
medianmouth=np.median(mouth) 

MARlist=np.array([])
#initially scroll_status is made 0 so that nose tip will be use for moving not scrolling
scroll_status=0
while True:
	try:
		initialTime=time.time()
		ret,image=cap.read()
		image=cv2.flip(image,1)
		#image=cv2.resize(image1,(680,480))
		blackimage = np.zeros((480,640,3),dtype = np.uint8)
		gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
		gray = clahe.apply(gray)
		rects=detector(gray,0)
		for (i,rect) in enumerate(rects):
			shape=predictor(gray,rect)
			shape=face_utils.shape_to_np(shape)
			[h,k]=shape[33]
			cv2.circle(image,(h,k),1,(255,0,0),2)
			cv2.line(image,(250,250),(h,k),(0,0,0),1)#line drawn joins center of circle with coordinate of facial landmark of nose
			cv2.circle(image,(250,250),50,(0,0,255),2)
			# Detection of eye with facial landmarks and then forming a convex hull on it.same for mouth
			leftEye = shape[lstart:lend]
			rightEye = shape[rstart:rend]
			#mouthroi = shape[48:59]
			#mouthroi2= shape[60:67]
			mouthroi = shape[mstart:mend]
			leftEyeHull = cv2.convexHull(leftEye)
			rightEyeHull = cv2.convexHull(rightEye)
			mouthHull = cv2.convexHull(mouthroi)
			#drawing contours
			cv2.drawContours(image,[mouthroi],-1,(0,255,0),1)
			cv2.drawContours(image,[leftEyeHull],-1,(0,255,0),1)
			cv2.drawContours(image,[rightEyeHull],-1,(0,255,0),1)
			cv2.drawContours(image,[mouthHull],-1,(0,255,0),1)
			#storing contour area in variable
			marea2 = cv2.contourArea(mouthHull)
			larea = cv2.contourArea(leftEyeHull)
			rarea = cv2.contourArea(rightEyeHull)
			MARlist = np.append(MARlist,[marea2]) # Appending the list at every iteration
			if len(MARlist) == 30: # till it reaches a size of 30 elements
				mar_avg = np.mean(MARlist)
				MARlist = np.array([]) # Resetting the MAR list
				if (int(mar_avg*100) > int(medianmouth*100)):
					if scroll_status == 0:
						scroll_status = 1
					else:
						scroll_status = 0
			if scroll_status == 0:#Enabling moving
				if((h-250)**2 + (k-250)**2 - 50**2 > 0):
					a = angle(shape[33]) # Calculates the angle
					if h > 250: # The below conditions set the conditions for the mouse to move and that too in any direction we desire it to move to.
						#time.sleep(0.03)
						pag.moveTo(pag.position()[0]+(10*np.cos(a)),pag.position()[1]+(10*np.sin(a)),duration = 0.01)
						cv2.putText(blackimage,"Moving",(0,250),font,1,(255,255,255),2,cv2.LINE_AA)
					else:
						#time.sleep(0.03)
						pag.moveTo(pag.position()[0]-(10*np.cos(a)),pag.position()[1]-(10*np.sin(a)),duration = 0.01)
						cv2.putText(blackimage,"Moving",(0,250),font,1,(255,255,255),2,cv2.LINE_AA)
			else: #Enabling scroll status
				cv2.putText(blackimage,'Scroll mode ON',(0,100),font,1,(255,255,255),2,cv2.LINE_AA)
				print('scroll mode is on')
				if k > 300: 
					pag.scroll(-1)
					cv2.putText(blackimage,"Scrolling Down",(0,300),font,1,(255,255,255),2,cv2.LINE_AA) 
				elif k < 200:
					pag.scroll(1)
					cv2.putText(blackimage,"Scrolling up",(0,300),font,1,(255,255,255),2,cv2.LINE_AA) 
				else:
					pass
		for (x,y) in shape:#printing facial land marks
			cv2.circle(image,(x,y),2,(0,255,0),1)
			
		
		right_eye=EAR(shape[42],shape[43],shape[44],shape[45],shape[46],shape[47])
		left_eye=EAR(shape[36],shape[37],shape[38],shape[39],shape[40],shape[41])
		real_diff=(left_eye-right_eye)*100
		area_diff=(larea- rarea)/marea2
		if  area_diff < medianleftarea and real_diff < (medianleft-3):
			pag.click(button='left')
			cv2.putText(blackimage,"Left Click",(0,300),font,1,(255,255,255),2,cv2.LINE_AA)
		elif area_diff > medianrightarea and real_diff > (medianright+3):
			pag.click(button='right')
			cv2.putText(blackimage,"Right Click",(0,300),font,1,(255,255,255),2,cv2.LINE_AA)
		else:
			pass

		
		
		finalTime=time.time()
		cv2.putText(blackimage,"FPS: "+str(int(1/(finalTime - initialTime))),(0,150),font,1,(255,255,255),2,cv2.LINE_AA)
		cv2.putText(blackimage,"Press Esc to abort",(0,200),font,1,(255,255,255),2,cv2.LINE_AA)
		res = np.vstack((image,blackimage))
		#cv2.namedWindow("output", cv2.WINDOW_NORMAL)
		#cv2.resizeWindow('output', 800,960)
		cv2.imshow('cursor control',res)
		if (cv2.waitKey(5) & 0xFF) == 27:
			break
	
	except ValueError:
		print('Paused')
		if (cv2.waitKey(5) & 0xFF) == 27:
			break




cap.release()
cv2.destroyAllWindows()


