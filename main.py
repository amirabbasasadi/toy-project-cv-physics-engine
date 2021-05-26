# A toy project with OpenCV, PyMunk and Mediapipe
import pymunk
import cv2
import numpy as np
import mediapipe as mp
mp_hands = mp.solutions.hands

# define the space for handling physics
space = pymunk.Space()
space.gravity = 0, -500

# define balls as dynamic bodies for physics engine
balls_radius = 12
balls = [(300 + np.random.uniform(-30, 30), 400 + 50*i + 0.5*i**2) for i in range(50)]
balls_body = [pymunk.Body(100.0,1666, body_type=pymunk.Body.DYNAMIC) for b in balls]
for i, ball in enumerate(balls_body): 
    balls_body[i].position = balls[i]
    shape = pymunk.Circle(balls_body[i], balls_radius)
    space.add(balls_body[i], shape)


# define fingers as kinematic bodies for physics engine
fingers_radius = 20
fingers = [pymunk.Body(10,1666, body_type=pymunk.Body.KINEMATIC) for i in range(21)]
for i, finger in enumerate(fingers):
    finger_shape = pymunk.Circle(fingers[i], fingers_radius)
    space.add(fingers[i], finger_shape)

# a few color for drawing balls
colors = [(219,152,52), (34, 126, 230), (182, 89, 155),
          (113, 204, 46), (94, 73, 52), (15, 196, 241),
          (60, 76, 231)]

# reading the video from webcam
cap = cv2.VideoCapture(0) 
with mp_hands.Hands(
    min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while True:
        success, image = cap.read()
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i, finger in enumerate(fingers):
                    # converting the coordinates
                    x = int(hand_landmarks.landmark[i].x * image.shape[1])
                    y = image.shape[0]-int(hand_landmarks.landmark[i].y * image.shape[0])
                    # update the velocity of balls
                    fingers[i].velocity = 14.0*(x - fingers[i].position[0]), 14.0*(y - fingers[i].position[1])
                    
	# getting the position of balls from physics engine and drawing
        for i, ball in enumerate(balls_body):
            xb = int(ball.position[0])
            yb = int(image.shape[0]-ball.position[1])
            cv2.circle(image, (xb, yb), balls_radius, colors[i%len(colors)], -1)
        
        # take a simulation step
        space.step(0.02)
        
        cv2.imshow("simulation", image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
