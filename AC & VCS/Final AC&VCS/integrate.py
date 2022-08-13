from handTracker import *
import cv2
import mediapipe as mp
import numpy as np
import random
import time

class ColorRect():
    def __init__(self, x1, y1, w, h, color, text='', alpha = 0.5):
        self.x = x1
        self.y = y1
        self.w = w
        self.h = h
        self.color = color
        self.text=text
        self.alpha = alpha
        
    
    def drawRect(self, img, text_color=(255,255,255), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8, thickness=2):
        #draw the box
        alpha = self.alpha
        bg_rec = img[self.y1 : self.y1 + self.h, self.x1 : self.x1 + self.w]
        white_rect = np.ones(bg_rec.shape, dtype=np.uint8)
        white_rect[:] = self.color
        res = cv2.addWeighted(bg_rec, alpha, white_rect, 1-alpha, 1.0)
        
        # Putting the image back to its position
        img[self.y1 : self.y1 + self.h, self.x1 : self.x1 + self.w] = res

        #put the letter
        tetx_size = cv2.getTextSize(self.text, fontFace, fontScale, thickness)
        text_pos = (int(self.x1 + self.w/2 - tetx_size[0][0]/2), int(self.y1 + self.h/2 + tetx_size[0][1]/2))
        cv2.putText(img, self.text,text_pos , fontFace, fontScale,text_color, thickness)


    def isOver(self,x1,y1):
        if (self.x1 + self.w > x1 > self.x) and (self.y1 + self.h> y >self.y1):
            return True
        return False


#initilize the habe detector
detector = HandTracker(detectionCon=0.8)



# creating canvas to draw on it
canvas = np.zeros((720,1280,3), np.uint8)

# define a previous point to be used with drawing a line
px,py = 0,0
#initial brush color
color = (255,0,0)
#####
brushSize = 5
eraserSize = 20
####

########### creating colors ########
# Colors button
colorsBtn = ColorRect(200, 0, 100, 100, (120,255,0), 'Colors')

colors = []
#random color
b = int(random.random()*255)-1
g = int(random.random()*255)
r = int(random.random()*255)
print(b,g,r)
colors.append(ColorRect(300,0,100,100, (b,g,r)))
#red
colors.append(ColorRect(400,0,100,100, (0,0,255)))
#blue
colors.append(ColorRect(500,0,100,100, (255,0,0)))
#green
colors.append(ColorRect(600,0,100,100, (0,255,0)))
#yellow
colors.append(ColorRect(700,0,100,100, (0,255,255)))
#erase (black)
colors.append(ColorRect(800,0,100,100, (0,0,0), "Eraser"))

#clear
clear = ColorRect(900,0,100,100, (100,100,100), "Clear")

########## pen sizes #######
pens = []
for i, penSize in enumerate(range(5,25,5)):
    pens.append(ColorRect(1100,50+100*i,100,100, (50,50,50), str(penSize)))

penBtn = ColorRect(1100, 0, 100, 50, color, 'Pen')

# white board button
boardBtn = ColorRect(50, 0, 100, 100, (255,255,0), 'Board')

#define a white board to draw on
whiteBoard = ColorRect(50, 120, 1020, 580, (255,255,255),alpha = 0.6)

coolingCounter = 20
hideBoard = True
hideColors = True
hidePenSizes = True

while True:

    if coolingCounter:
        coolingCounter -=1
        #print(coolingCounter)
    
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.resize(frame, (1280, 720))
    frame = cv2.flip(frame, 1)

    detector.findHands(frame)
    positions = detector.getPostion(frame, draw=False)
    upFingers = detector.getUpFingers(frame)

    if upFingers:
        x1, y1 = positions[8][0], positions[8][1]
        if upFingers[1] and not whiteBoard.isOver(x1, y1):
            px, py = 0, 0

            ##### pen sizes ######
            if not hidePenSizes:
                for pen in pens:
                    if pen.isOver(x1, y1):
                        brushSize = int(pen.text)
                        pen.alpha = 0
                    else:
                        pen.alpha = 0.5

            ####### chose a color for drawing #######
            if not hideColors:
                for cb in colors:
                    if cb.isOver(x1, y1):
                        color = cb.color
                        cb.alpha = 0
                    else:
                        cb.alpha = 0.5

                #Clear 
                if clear.isOver(x1, y1):
                    clear.alpha = 0
                    canvas = np.zeros((720,1280,3), np.uint8)
                else:
                    clear.alpha = 0.5
            
            # color button
            if colorsBtn.isOver(x1, y1) and not coolingCounter:
                coolingCounter = 10
                colorsBtn.alpha = 0
                hideColors = False if hideColors else True
                colorsBtn.text = 'Colors' if hideColors else 'Hide'
            else:
                colorsBtn.alpha = 0.5
            
            # Pen size button
            if penBtn.isOver(x1, y1) and not coolingCounter:
                coolingCounter = 10
                penBtn.alpha = 0
                hidePenSizes = False if hidePenSizes else True
                penBtn.text = 'Pen' if hidePenSizes else 'Hide'
            else:
                penBtn.alpha = 0.5

            
            #white board button
            if boardBtn.isOver(x1, 1) and not coolingCounter:
                coolingCounter = 10
                boardBtn.alpha = 0
                hideBoard = False if hideBoard else True
                boardBtn.text = 'Board' if hideBoard else 'Hide'

            else:
                boardBtn.alpha = 0.5
            
            
            

        elif upFingers[1] and not upFingers[2]:
            if whiteBoard.isOver(x1, y1) and not hideBoard:
                #print('index finger is up')
                cv2.circle(frame, positions[8], brushSize, color,-1)
                #drawing on the canvas
                if px == 0 and py == 0:
                    px, py = positions[8]
                if color == (0,0,0):
                    cv2.line(canvas, (px,py), positions[8], color, eraserSize)
                else:
                    cv2.line(canvas, (px,py), positions[8], color,brushSize)
                px, py = positions[8]
        
        else:
            px, py = 0, 0
        
    # put colors button
    colorsBtn.drawRect(frame)
    cv2.rectangle(frame, (colorsBtn.x1, colorsBtn.y1), (colorsBtn.x1 +colorsBtn.w, colorsBtn.y1+colorsBtn.h), (255,255,255), 2)

    # put white board buttin
    boardBtn.drawRect(frame)
    cv2.rectangle(frame, (boardBtn.x1, boardBtn.y1), (boardBtn.x1 +boardBtn.w, boardBtn.y1+boardBtn.h), (255,255,255), 2)

    #put the white board on the frame
    if not hideBoard:       
        whiteBoard.drawRect(frame)
        ########### moving the draw to the main image #########
        canvasGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, imgInv = cv2.threshold(canvasGray, 20, 255, cv2.THRESH_BINARY_INV)
        imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, imgInv)
        frame = cv2.bitwise_or(frame, canvas)


    ########## pen colors' boxes #########
    if not hideColors:
        for c in colors:
            c.drawRect(frame)
            cv2.rectangle(frame, (c.x1, c.y1), (c.x1 +c.w, c.y1+c.h), (255,255,255), 2)

        clear.drawRect(frame)
        cv2.rectangle(frame, (clear.x1, clear.y1), (clear.x1 +clear.w, clear.y1+clear.h), (255,255,255), 2)


    ########## brush size boxes ######
    penBtn.color = color
    penBtn.drawRect(frame)
    cv2.rectangle(frame, (penBtn.x1, penBtn.y1), (penBtn.x1 +penBtn.w, penBtn.y1+penBtn.h), (255,255,255), 2)
    if not hidePenSizes:
        for pen in pens:
            pen.drawRect(frame)
            cv2.rectangle(frame, (pen.x1, pen.y1), (pen.x1 +pen.w, pen.y+pen.h), (255,255,255), 2)


    
#shapes
#contants
ml = 150
max_x, max_y = 250+ml, 50
curr_tool = "select tool"
time_init = True
rad = 40
var_inits = False
thick = 4
prevx, prevy = 0,0

#get tools function
def getTool(x):
	if x < 50 + ml:
		return "line"

	elif x<100 + ml:
		return "rectangle"

	elif x < 150 + ml:
		return"draw"

	elif x<200 + ml:
		return "circle"

	else:
		return "erase"

def index_raised(yi, y9):
	if (y9 - yi) > 40:
		return True

	return False



hands = mp.solutions.hands
hand_landmark = hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1)
draw = mp.solutions.drawing_utils


# drawing tools
tools = cv2.imread("tools.png")
tools = tools.astype('uint8')

mask = np.ones((480, 640))*255
mask = mask.astype('uint8')
'''
tools = np.zeros((max_y+5, max_x+5, 3), dtype="uint8")
cv2.rectangle(tools, (0,0), (max_x, max_y), (0,0,255), 2)
cv2.line(tools, (50,0), (50,50), (0,0,255), 2)
cv2.line(tools, (100,0), (100,50), (0,0,255), 2)
cv2.line(tools, (150,0), (150,50), (0,0,255), 2)
cv2.line(tools, (200,0), (200,50), (0,0,255), 2)
'''

cap = cv2.VideoCapture(0)
while True:
	_, frame = cap.read()
	frame = cv2.flip(frame, 1)

	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	op = hand_landmark.process(rgb)

	if op.multi_hand_landmarks:
		for i in op.multi_hand_landmarks:
			draw.draw_landmarks(frame, i, hands.HAND_CONNECTIONS)
			x, y = int(i.landmark[8].x*640), int(i.landmark[8].y*480)

			if x < max_x and y < max_y and x > ml:
				if time_init:
					ctime = time.time()
					time_init = False
				ptime = time.time()

				cv2.circle(frame, (x, y), rad, (0,255,255), 2)
				rad -= 1

				if (ptime - ctime) > 0.8:
					curr_tool = getTool(x)
					print("your current tool set to : ", curr_tool)
					time_init = True
					rad = 40

			else:
				time_init = True
				rad = 40

			if curr_tool == "draw":
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
				y9  = int(i.landmark[9].y*480)

				if index_raised(yi, y9):
					cv2.line(mask, (prevx, prevy), (x, y), 0, thick)
					prevx, prevy = x, y

				else:
					prevx = x
					prevy = y



			elif curr_tool == "line":
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
				y9  = int(i.landmark[9].y*480)

				if index_raised(yi, y9):
					if not(var_inits):
						xii, yii = x, y
						var_inits = True

					cv2.line(frame, (xii, yii), (x, y), (50,152,255), thick)

				else:
					if var_inits:
						cv2.line(mask, (xii, yii), (x, y), 0, thick)
						var_inits = False

			elif curr_tool == "rectangle":
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
				y9  = int(i.landmark[9].y*480)

				if index_raised(yi, y9):
					if not(var_inits):
						xii, yii = x, y
						var_inits = True

					cv2.rectangle(frame, (xii, yii), (x, y), (0,255,255), thick)

				else:
					if var_inits:
						cv2.rectangle(mask, (xii, yii), (x, y), 0, thick)
						var_inits = False

			elif curr_tool == "circle":
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
				y9  = int(i.landmark[9].y*480)

				if index_raised(yi, y9):
					if not(var_inits):
						xii, yii = x, y
						var_inits = True

					cv2.circle(frame, (xii, yii), int(((xii-x)**2 + (yii-y)**2)**0.5), (255,255,0), thick)

				else:
					if var_inits:
						cv2.circle(mask, (xii, yii), int(((xii-x)**2 + (yii-y)**2)**0.5), (0,255,0), thick)
						var_inits = False

			elif curr_tool == "erase":
				xi, yi = int(i.landmark[12].x*640), int(i.landmark[12].y*480)
				y9  = int(i.landmark[9].y*480)

				if index_raised(yi, y9):
					cv2.circle(frame, (x, y), 30, (0,0,0), -1)
					cv2.circle(mask, (x, y), 30, 255, -1)



	op = cv2.bitwise_and(frame, frame, mask=mask)
	frame[:, :, 1] = op[:, :, 1]
	frame[:, :, 2] = op[:, :, 2]

	frame[:max_y, ml:max_x] = cv2.addWeighted(tools, 0.7, frame[:max_y, ml:max_x], 0.3, 0)


    
