import cv2
import mediapipe as mp
import random
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Game variables
player_score = 0
computer_score = 0
computer_choice = ""
result = ""
last_prediction_time = 0

# Gesture classification function
def classify_gesture(landmarks):
    if not landmarks:
        return None
    
    # Get thumb, index, middle, ring, pinky tips and PIP joints
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]
    
    # Thumb direction (rock detection)
    thumb_base = landmarks[2]
    thumb_direction = thumb_tip.x < thumb_base.x  # Thumb tucked in
    
    # Finger tips above PIP (extended)
    index_pip = landmarks[6].y
    middle_pip = landmarks[10].y
    ring_pip = landmarks[14].y
    pinky_pip = landmarks[18].y
    
    fingers_up = [
        index_tip.y < index_pip,
        middle_tip.y < middle_pip,
        ring_tip.y < ring_pip,
        pinky_tip.y < pinky_pip
    ]
    
    # Classify gesture
    if thumb_direction and not any(fingers_up):
        return "ROCK"
    elif not thumb_direction and sum(fingers_up) >= 3:
        return "PAPER"
    elif sum(fingers_up) == 2:  # Index + middle for scissors
        return "SCISSORS"
    return None

# Main game loop
cap = cv2.VideoCapture(0)

print("=== GESTURE ROCK PAPER SCISSORS ===")
print("Show gesture for 2 seconds to play!")
print("Controls: Close window to quit")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    image_copy = image.copy()
    
    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image_copy, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Classify gesture
            landmarks = hand_landmarks.landmark
            gesture = classify_gesture(landmarks)
            
            if gesture:
                current_time = time.time()
                if current_time - last_prediction_time > 2.0:  # 2 second hold
                    player_choice = gesture
                    computer_choice = random.choice(["ROCK", "PAPER", "SCISSORS"])
                    last_prediction_time = current_time
                    
                    # Determine winner
                    if player_choice == computer_choice:
                        result = "TIE!"
                    elif (player_choice == "ROCK" and computer_choice == "SCISSORS") or \
                         (player_choice == "PAPER" and computer_choice == "ROCK") or \
                         (player_choice == "SCISSORS" and computer_choice == "PAPER"):
                        result = "YOU WIN! 🎉"
                        player_score += 1
                    else:
                        result = "COMPUTER WINS! 😢"
                        computer_score += 1
    
    # Display info
    cv2.rectangle(image_copy, (10, 10), (400, 80), (0, 0, 0), -1)
    cv2.putText(image_copy, f"Player: {player_score} | Comp: {computer_score}", 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if result:
        cv2.putText(image_copy, f"You: {player_choice} | Comp: {computer_choice}", 
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(image_copy, result, (450, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow('Gesture Rock Paper Scissors', image_copy)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Final Score - You: {player_score} | Computer: {computer_score}")
