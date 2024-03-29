# CURSOR-CONTROL-USING-FACE-GESTURES

## OVERVIEW

Computer Vision is used for catering the needs of the users who are invalid in the sense that they are not able to perform basic operations on the computer such as the use of mouse. Without the use of any physical help, the user can now be able to use the mouse using simple facial gestures such as blinking. Blinking of left or right eye will initiate a left or right click respectively.

Many of the products related to this project are present in the market and are often expensive or require some external hardware for configuration. In this project, we aim at designing a similar product which does not require any external hardware and is open – sourced.

<img src="images/LANDMARKS.jpg">

A face is divided into 68 distinct landmarks, rather than taking the entire face into account, which makes the working algorithm very fast. Exploiting the orientation of the points on the eye, we use the concept of Eye Aspect Ratio, which is used to determine whether the eye is open or closed.

<img src="images/EAR.jpg">

For scrolling on the page, we have used the concept of Mouth Aspect Ratio, which is just used to determine whether the mouth is open or closed. Opening the mouth for about 5 seconds will activate the scrolling mode and the mouse which used to move up and down will now scroll up and down. For deactivating the scroll one, the user can do the same action again.

<img src="images/MAR.jpg">

For moving the mouse cursor across the screen, the user must move the nose out of a reference circle. In whatever direction the user keeps their nose with respect to the centre of the circle, the mouse will move in that direction. Keeping the nose tip in the reference circle will stop the moving cursor.

The above code was tested and run on Intel® Core™ i5-8250U CPU @ 1.60GHz × 8. All the code was written on gedit file in Ubuntu version 18.04.02 LTS

## REQUIREMENTS
* numpy==1.16.4 
* scipy==1.3.0 
* matplotlib==3.1.0 
* dlib==19.17.0 
* imutils==0.5.2 
* opencv-python==4.1.0
* pyautogui==0.9.4

## SAMPLE IMAGE DURING CALIBRATION

![image1](https://user-images.githubusercontent.com/50697244/61187022-f1dc7c80-a689-11e9-8e4f-c91bf16dfacf.png)

## SAMPLE IMAGE WHILE USING ALGORITHM 

![image2](https://user-images.githubusercontent.com/50697244/61187039-4f70c900-a68a-11e9-974d-9ba45d0ecc90.png)

Here user is going to select option from drop down menu.The black window will show the current frames per second and shows whatever action we are doing, for example, moving the mouse, left click, right click and also shows whether the scroll mode is on or not.

#### HOW CODE WORKS?
[Video_On_GoogleDrive](https://drive.google.com/file/d/1VU5pctXON-NsvULlY1GlcKG_zVksxDda/view?usp=sharing)

## TEAM MEMBERS
* 1.ARIHANT GAUR
* 2.NILAKSHI REKHAWAR
* 3.AKSHATA KINAGE
* 4.SHUBHAN RUKUMANGAD

## MENTORS
* 1.Rohit Lal
* 2.Shubham Chauhan
* 3.Abhay Khandelwal
