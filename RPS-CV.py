import random
import time
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np

# The mechanistic justification for this step is: Updating RPS-CV to utilize the PyTorch model instead of randomness, creating an adaptive AI loop.

class RPSModel(nn.Module):
    def __init__(self):
        super(RPSModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)
        self.fc = nn.Linear(16, 3)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = RPSModel()
try:
    model.load_state_dict(torch.load('rps_lstm_model.pth'))
    model.eval()
except Exception as e:
    print(f"Model not found or error loading: {e}. Random actions will be used initially.")

def overlayPNG(imgBack, imgFront, pos=[0, 0]):
    hf, wf, cf = imgFront.shape
    hb, wb, cb = imgBack.shape
    *_, mask = cv2.split(imgFront)
    maskBGRA = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)
    maskBGR = maskBGRA[:, :, :3]
    imgRGBA = cv2.bitwise_and(imgFront, maskBGRA)
    imgRGB = imgRGBA[:, :, :3]

    y1, y2 = max(0, pos[1]), min(hb, pos[1] + hf)
    x1, x2 = max(0, pos[0]), min(wb, pos[0] + wf)
    y1f, y2f = max(0, -pos[1]), min(hf, hb - pos[1])
    x1f, x2f = max(0, -pos[0]), min(wf, wb - pos[0])

    if y1 < y2 and x1 < x2:
        imgBack[y1:y2, x1:x2] = cv2.bitwise_and(imgBack[y1:y2, x1:x2], cv2.bitwise_not(maskBGR[y1f:y2f, x1f:x2f])) + imgRGB[y1f:y2f, x1f:x2f]
    return imgBack

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

timer = 0
stateResult = False
startGame = False
scores = [0, 0]  # [AI, Player]
initialTime = 0
player_history = []
imgAI = None

while True:
    imgBG = cv2.imread("Resources/BG.png")
    if imgBG is None:
        break
    success, img = cap.read()
    if not success:
        break

    imgScaled = cv2.resize(img, (0, 0), None, 0.875, 0.875)
    imgScaled = imgScaled[:, 80:480]

    imgRGB = cv2.cvtColor(imgScaled, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(imgScaled, handLms, mp_hands.HAND_CONNECTIONS)

    if startGame:
        if stateResult is False:
            timer = time.time() - initialTime
            cv2.putText(imgBG, str(int(timer)), (605, 435), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 255), 4)

            if timer > 3:
                stateResult = True
                timer = 0

                if results.multi_hand_landmarks:
                    playerMove = None
                    handLms = results.multi_hand_landmarks[0]
                    lmList = []
                    for id, lm in enumerate(handLms.landmark):
                        h, w, c = imgScaled.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        lmList.append([id, cx, cy])
                    
                    fingers = []
                    # Thumb
                    if lmList[4][1] > lmList[3][1]:
                        fingers.append(1)
                    else:
                        fingers.append(0)
                    # 4 Fingers
                    for id in range(8, 21, 4):
                        if lmList[id][2] < lmList[id - 2][2]:
                            fingers.append(1)
                        else:
                            fingers.append(0)

                    if fingers == [0, 0, 0, 0, 0]:
                        playerMove = 1
                    elif fingers == [1, 1, 1, 1, 1]:
                        playerMove = 2
                    elif fingers == [0, 1, 1, 0, 0]:
                        playerMove = 3
                    
                    if playerMove is not None:
                        # Predict next move
                        if len(player_history) >= 5:
                            seq = player_history[-5:]
                            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
                            with torch.no_grad():
                                pred = model(seq_tensor)
                                predicted_player_move = torch.argmax(pred, dim=1).item() + 1
                            aiMove = (predicted_player_move % 3) + 1
                        else:
                            aiMove = random.randint(1, 3)

                        randomNumber = aiMove
                        player_history.append(playerMove)
                        
                        imgAI = cv2.imread(f'Resources/{randomNumber}.png', cv2.IMREAD_UNCHANGED)
                        if imgAI is not None:
                            imgBG = overlayPNG(imgBG, imgAI, (149, 310))

                        # Player Wins
                        if (playerMove == 1 and randomNumber == 3) or \
                                (playerMove == 2 and randomNumber == 1) or \
                                (playerMove == 3 and randomNumber == 2):
                            scores[1] += 1
                        # AI Wins
                        elif (playerMove == 3 and randomNumber == 1) or \
                                (playerMove == 1 and randomNumber == 2) or \
                                (playerMove == 2 and randomNumber == 3):
                            scores[0] += 1
                else:
                    randomNumber = None
                    imgAI = None

    imgBG[234:654, 795:1195] = imgScaled

    if stateResult:
        if imgAI is not None:
            imgBG = overlayPNG(imgBG, imgAI, (149, 310))

    cv2.putText(imgBG, str(scores[0]), (410, 215), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 6)
    cv2.putText(imgBG, str(scores[1]), (1112, 215), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 6)

    cv2.imshow("BG", imgBG)
    key = cv2.waitKey(1)
    if key == ord('s'):
        startGame = True
        initialTime = time.time()
        stateResult = False
    elif key == ord('q'):
        break