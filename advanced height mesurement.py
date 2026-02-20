
import cv2 as cv
import mediapipe as mp
import numpy as np
import time
import pyttsx3. 

# Initialize Text-to-Speech
try:
    engine = pyttsx3.init()
    def speak(text):
        print(f"SPEAKING: {text}")
        engine.say(text)
        engine.runAndWait()
except Exception as e:
    print(f"TTS Error: {e}")
    def speak(text):
        print(f"SPEAK: {text}")

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)

face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open camera
cap = cv.VideoCapture(0)
if not cap.isOpened():
    cap = cv.VideoCapture(1)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

# Constants
AVG_FACE_HEIGHT_CM = 20  # Average face height from chin to forehead
AVG_EYE_TO_EYE_CM = 6.5  # Average distance between eyes

# Storage
calibration_ratio = None
calibration_method = None
current_height = 0
height_announced = False
measurements = []
debug_info = []
calibration_frames = []
calibration_complete_time = None
stage = "CALIBRATE"  # CALIBRATE or MEASURE

print("\n" + "="*50)
print("HEIGHT MEASUREMENT SYSTEM")
print("="*50)
print("\nINSTRUCTIONS:")
print("1. First CALIBRATION: Show your FACE close to camera")
print("2. Then MEASUREMENT: STEP BACK so full body is visible")
print("3. Stand straight with feet clearly visible")
print("4. Press 'q' to quit\n")

while True:
    success, img = cap.read()
    if not success:
        break
    
    # Flip image horizontally for mirror effect
    img = cv.flip(img, 1)
    h, w, _ = img.shape
    
    # Convert to RGB
    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    # Process face for calibration
    face_results = face_mesh.process(img_rgb)
    
    # Process pose for body
    pose_results = pose.process(img_rgb)
    
    # Clear debug info each frame
    debug_info = []
    
    # ===== CALIBRATION PHASE =====
    if stage == "CALIBRATE" and face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            # Method 1: Face height calibration
            y_coords = [lm.y * h for lm in face_landmarks.landmark]
            face_h_px = max(y_coords) - min(y_coords)
            
            # Method 2: Eye distance calibration (more stable)
            # Get left eye (indices 33, 133) and right eye (362, 263)
            left_eye_indices = [33, 133]
            right_eye_indices = [362, 263]
            
            left_eye_center = np.mean([(face_landmarks.landmark[i].x * w, 
                                       face_landmarks.landmark[i].y * h) 
                                      for i in left_eye_indices], axis=0)
            right_eye_center = np.mean([(face_landmarks.landmark[i].x * w, 
                                        face_landmarks.landmark[i].y * h) 
                                       for i in right_eye_indices], axis=0)
            
            eye_distance_px = np.linalg.norm(left_eye_center - right_eye_center)
            
            if face_h_px > 30 and eye_distance_px > 10:
                # Store multiple calibration frames for averaging
                calibration_frames.append({
                    'face_h': face_h_px,
                    'eye_dist': eye_distance_px,
                    'timestamp': time.time()
                })
                
                # Keep last 10 frames
                if len(calibration_frames) > 10:
                    calibration_frames.pop(0)
                
                # Show calibration progress
                cv.putText(img, f"Calibration frames: {len(calibration_frames)}/10", 
                          (20, 200), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Draw face landmarks
                mp_draw.draw_landmarks(img, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
                
                # Draw eye distance
                cv.line(img, tuple(left_eye_center.astype(int)), 
                       tuple(right_eye_center.astype(int)), (0, 255, 0), 2)
                
                # After collecting enough frames, compute average calibration
                if len(calibration_frames) >= 8 and calibration_complete_time is None:
                    avg_face_h = np.mean([f['face_h'] for f in calibration_frames])
                    avg_eye_dist = np.mean([f['eye_dist'] for f in calibration_frames])
                    
                    # Use both methods and average them
                    ratio_face = AVG_FACE_HEIGHT_CM / avg_face_h
                    ratio_eye = AVG_EYE_TO_EYE_CM / avg_eye_dist
                    
                    calibration_ratio = (ratio_face + ratio_eye) / 2
                    calibration_complete_time = time.time()
                    stage = "MEASURE"
                    
                    print(f"\n✓ CALIBRATION COMPLETE!")
                    print(f"  Face height: {avg_face_h:.0f} pixels")
                    print(f"  Eye distance: {avg_eye_dist:.0f} pixels")
                    print(f"  Ratio: {calibration_ratio:.4f} cm/pixel")
                    speak("Calibration complete. Now show your full body.")
    
    # ===== MEASUREMENT PHASE =====
    if stage == "MEASURE" and pose_results.pose_landmarks:
        # Draw pose landmarks
        mp_draw.draw_landmarks(
            img, 
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
            mp_draw.DrawingSpec(color=(0,0,255), thickness=2)
        )
        
        # Get all landmarks
        landmarks = pose_results.pose_landmarks.landmark
        
        # TOP POINT: Use nose or top of head (index 0 for nose)
        top_point = None
        top_y = float('inf')
        
        # Check nose (most reliable)
        nose = landmarks[0]
        if nose.visibility > 0.7:
            top_y = nose.y * h
            top_point = (w//2, int(top_y))
            debug_info.append(f"Nose visible: {nose.visibility:.2f}")
        
        # Also check eye and ear for better top estimation
        left_eye = landmarks[2]  # Left eye
        right_eye = landmarks[5]  # Right eye
        left_ear = landmarks[7]   # Left ear
        right_ear = landmarks[8]  # Right ear
        
        if left_eye.visibility > 0.7:
            top_y = min(top_y, left_eye.y * h)
        if right_eye.visibility > 0.7:
            top_y = min(top_y, right_eye.y * h)
        if left_ear.visibility > 0.7:
            top_y = min(top_y, left_ear.y * h)
        if right_ear.visibility > 0.7:
            top_y = min(top_y, right_ear.y * h)
        
        # BOTTOM POINT: Try multiple foot/ankle points
        bottom_y = 0
        bottom_point = None
        foot_detected = False
        
        # Check feet and ankles
        foot_indices = [
            (31, "Left Foot"),   # Left foot index
            (32, "Right Foot"),  # Right foot index
            (27, "Left Ankle"),  # Left ankle
            (28, "Right Ankle"), # Right ankle
            (29, "Left Heel"),   # Left heel
            (30, "Right Heel"),  # Right heel
        ]
        
        for idx, name in foot_indices:
            if idx < len(landmarks):
                point = landmarks[idx]
                if point.visibility > 0.7:
                    y_pos = point.y * h
                    if y_pos > bottom_y:
                        bottom_y = y_pos
                        bottom_point = (w//2, int(y_pos))
                        foot_detected = True
                        debug_info.append(f"{name}: {point.visibility:.2f}")
        
        # If feet not visible, try to estimate from hips
        if not foot_detected:
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            left_knee = landmarks[25]
            right_knee = landmarks[26]
            
            if (left_hip.visibility > 0.7 and right_hip.visibility > 0.7 and
                left_knee.visibility > 0.7 and right_knee.visibility > 0.7):
                
                hip_y = (left_hip.y * h + right_hip.y * h) / 2
                knee_y = (left_knee.y * h + right_knee.y * h) / 2
                
                # Estimate foot position: foot_y = knee_y + (knee_y - hip_y)
                estimated_foot_y = knee_y + (knee_y - hip_y)
                bottom_y = estimated_foot_y
                bottom_point = (w//2, int(estimated_foot_y))
                debug_info.append("Using estimated foot position")
        
        # Calculate height if we have both points
        if top_point and bottom_point and bottom_y > top_y:
            height_px = bottom_y - top_y
            debug_info.append(f"Height pixels: {height_px:.0f}")
            
            # Draw measurement line (YELLOW LINE as in your photo)
            cv.line(img, top_point, bottom_point, (0, 255, 255), 4)
            cv.circle(img, top_point, 8, (0, 255, 0), -1)
            cv.circle(img, bottom_point, 8, (0, 0, 255), -1)
            
            # Add labels
            cv.putText(img, "TOP", (top_point[0] + 15, top_point[1] - 10),
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv.putText(img, "BOTTOM", (bottom_point[0] + 15, bottom_point[1] - 10),
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Convert to cm
            if calibration_ratio is not None:
                height_cm = height_px * calibration_ratio
                debug_info.append(f"Raw height: {height_cm:.1f}cm")
                
                # Validate (reasonable height range)
                if 100 < height_cm < 250:
                    measurements.append(height_cm)
                    debug_info.append(f"Valid! Added to measurements")
                    
                    # Keep last 15 measurements
                    if len(measurements) > 15:
                        measurements.pop(0)
                    
                    # Calculate moving average
                    if len(measurements) >= 5:
                        # Remove outliers
                        sorted_meas = sorted(measurements[-10:])
                        if len(sorted_meas) > 3:
                            trimmed = sorted_meas[1:-1]  # Remove highest and lowest
                            current_height = sum(trimmed) / len(trimmed)
                        else:
                            current_height = sum(measurements) / len(measurements)
                        
                        # Announce when stable
                        if not height_announced and len(measurements) >= 10:
                            speak(f"Your height is {current_height:.0f} centimeters")
                            height_announced = True
                            print(f"\n✓ HEIGHT ANNOUNCED: {current_height:.1f} cm")
    
    # ===== DISPLAY ON SCREEN =====
    
    # Create semi-transparent background for text
    overlay = img.copy()
    cv.rectangle(overlay, (10, 10), (500, 400), (0, 0, 0), -1)
    cv.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    y_pos = 40
    
    # Title
    cv.putText(img, "HEIGHT MEASUREMENT SYSTEM", (20, y_pos), 
              cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # Stage indicator
    if stage == "CALIBRATE":
        cv.putText(img, "STAGE 1: CALIBRATION", (20, y_pos), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y_pos += 35
        cv.putText(img, "Show your face close to camera", (20, y_pos), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += 30
        
        if face_results.multi_face_landmarks:
            cv.putText(img, "✓ Face detected", (20, y_pos), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_pos += 30
            cv.putText(img, f"Calibration frames: {len(calibration_frames)}/10", (20, y_pos), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv.putText(img, "⚠️ No face detected", (20, y_pos), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    else:  # MEASURE stage
        cv.putText(img, "STAGE 2: MEASUREMENT", (20, y_pos), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_pos += 35
        cv.putText(img, "Step back - show full body", (20, y_pos), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_pos += 30
        
        if current_height > 0:
            # Show height
            cv.putText(img, "YOUR HEIGHT:", (20, y_pos), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y_pos += 50
            
            # Large height display
            cv.putText(img, f"{current_height:.1f} cm", (20, y_pos), 
                      cv.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 5)
            y_pos += 70
            
            # Feet/inches
            inches = current_height / 2.54
            feet = int(inches // 12)
            remainder = int(inches % 12)
            cv.putText(img, f"{feet}'{remainder}\"", (20, y_pos), 
                      cv.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 255), 4)
            y_pos += 50
            
            # Progress
            cv.putText(img, f"Readings: {len(measurements)}/15", (20, y_pos), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y_pos += 25
            
            if height_announced:
                cv.putText(img, "✓ Height announced", (20, y_pos), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            else:
                cv.putText(img, f"⏳ Stabilizing... {len(measurements)}/10", (20, y_pos), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        else:
            cv.putText(img, "⚠️ Waiting for body detection", (20, y_pos), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_pos += 30
            if pose_results.pose_landmarks:
                if foot_detected:
                    cv.putText(img, "✓ Full body detected", (20, y_pos), 
                              cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv.putText(img, "⚠️ Feet not visible - step back", (20, y_pos), 
                              cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv.putText(img, "⚠️ No body detected", (20, y_pos), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Instructions
    cv.putText(img, "Press 'q' to quit", (w-150, 30), 
              cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Display
    cv.imshow("Height Measurement", img)
    
    # Quit on 'q'
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Show final height
if current_height > 0:
    inches = current_height / 2.54
    feet = int(inches // 12)
    remainder = int(inches % 12)
    print(f"\n{'='*50}")
    print(f"FINAL HEIGHT: {current_height:.1f} cm ({feet}'{remainder}\")")
    print(f"Measurements used: {len(measurements)}")
    print(f"{'='*50}")
    speak(f"Final height is {current_height:.1f} centimeters")
else:
    print("\nNo height measurement obtained")

cap.release()
cv.destroyAllWindows()
