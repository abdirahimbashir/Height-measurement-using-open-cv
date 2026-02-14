import cv2 as cv
import mediapipe as mp
import numpy as np
import time
import pyttsx3

# Initialize Text-to-Speech
try:
    engine = pyttsx3.init()
    def speak(text):
        print(f"SPEAKING: {text}")  # Debug print
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
    min_tracking_confidence=0.5
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

cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

# Constants
FACE_HEIGHT_CM = 20  # Average face height

# Storage
calibration_ratio = None
current_height = 0
height_announced = False
measurements = []
debug_info = []

print("\n" + "="*50)
print("HEIGHT MEASUREMENT SYSTEM")
print("="*50)
print("\nINSTRUCTIONS:")
print("1. First show your FACE close to camera")
print("2. Then STEP BACK so full body is visible")
print("3. Stand straight with feet visible")
print("4. Press 'q' to quit\n")

while True:
    success, img = cap.read()
    if not success:
        break
    
    # Flip image horizontally
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
    
    # Draw pose if detected
    if pose_results.pose_landmarks:
        mp_draw.draw_landmarks(
            img, 
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )
        
        # Get all landmarks
        landmarks = pose_results.pose_landmarks.landmark
        
        # Get nose (top reference)
        nose = landmarks[0]
        nose_y = nose.y * h
        debug_info.append(f"Nose visible: {nose.visibility:.2f}")
        
        # Try multiple points for bottom detection (in order of preference)
        bottom_points = []
        
        # Try feet/ankles in order
        foot_indices = [
            (31, "Left Foot"),   # Left foot
            (32, "Right Foot"),  # Right foot
            (29, "Left Heel"),   # Left heel
            (30, "Right Heel"),  # Right heel
            (27, "Left Ankle"),  # Left ankle
            (28, "Right Ankle"), # Right ankle
        ]
        
        for idx, name in foot_indices:
            if idx < len(landmarks):
                point = landmarks[idx]
                if point.visibility > 0.5:
                    bottom_points.append((point.y * h, name))
                    debug_info.append(f"{name}: {point.visibility:.2f}")
        
        # Also try using hip to estimate if feet not visible
        if not bottom_points:
            # Use hip points to estimate
            left_hip = landmarks[23]
            right_hip = landmarks[24]
            if left_hip.visibility > 0.5 and right_hip.visibility > 0.5:
                hip_y = (left_hip.y * h + right_hip.y * h) / 2
                # Rough estimate: total height ≈ 2.3 × hip height
                estimated_bottom = hip_y + (hip_y - nose_y) * 1.3
                bottom_points.append((estimated_bottom, "Estimated from hips"))
                debug_info.append("Using hip estimation")
        
        if bottom_points and nose.visibility > 0.5:
            # Get the lowest point (closest to feet)
            bottom_points.sort(reverse=True)  # Sort by y coordinate (largest = lowest)
            lowest_y, point_name = bottom_points[0]
            
            # Calculate height in pixels
            height_px = lowest_y - nose_y
            debug_info.append(f"Height pixels: {height_px:.0f}")
            
            # If we have calibration, convert to cm
            if calibration_ratio is not None:
                height_cm = height_px * calibration_ratio
                debug_info.append(f"Raw height: {height_cm:.1f}cm")
                
                # Validate (reasonable height range)
                if 100 < height_cm < 250:
                    measurements.append(height_cm)
                    debug_info.append(f"Valid! Added to measurements")
                    
                    # Keep last 10 measurements
                    if len(measurements) > 10:
                        measurements.pop(0)
                    
                    # Calculate average
                    if len(measurements) >= 3:
                        current_height = sum(measurements) / len(measurements)
                        
                        # Draw measurement line
                        cv.line(img, (w//2, int(nose_y)), (w//2, int(lowest_y)), (0, 255, 255), 3)
                        cv.circle(img, (w//2, int(nose_y)), 8, (0, 255, 0), -1)
                        cv.circle(img, (w//2, int(lowest_y)), 8, (0, 0, 255), -1)
                        
                        # Add labels
                        cv.putText(img, "Top", (w//2 + 10, int(nose_y) - 10),
                                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv.putText(img, f"Bottom ({point_name[:10]})", (w//2 + 10, int(lowest_y) - 10),
                                  cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                        
                        # Announce height when stable
                        if not height_announced and len(measurements) >= 8:
                            speak(f"Your height is {current_height:.0f} centimeters")
                            height_announced = True
                            print(f"\n✓ HEIGHT ANNOUNCED: {current_height:.1f} cm")
                else:
                    debug_info.append(f"Invalid height: {height_cm:.1f}cm")
    
    # Face calibration
    if face_results.multi_face_landmarks and calibration_ratio is None:
        for face_landmarks in face_results.multi_face_landmarks:
            # Get face height
            y_coords = [lm.y * h for lm in face_landmarks.landmark]
            face_h_px = max(y_coords) - min(y_coords)
            
            if face_h_px > 20:
                calibration_ratio = FACE_HEIGHT_CM / face_h_px
                print(f"\n✓ CALIBRATION COMPLETE!")
                print(f"  Face height: {face_h_px:.0f} pixels")
                print(f"  Ratio: {calibration_ratio:.4f} cm/pixel")
                speak("Calibration complete. Now show your full body.")
    
    # ===== DISPLAY ON SCREEN =====
    
    # Create semi-transparent background
    overlay = img.copy()
    cv.rectangle(overlay, (10, 10), (450, 350), (0, 0, 0), -1)
    cv.addWeighted(overlay, 0.7, img, 0.3, 0, img)
    
    y_pos = 40
    
    # Title
    cv.putText(img, "HEIGHT MEASUREMENT", (20, y_pos), 
              cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_pos += 40
    
    # Calibration status
    if calibration_ratio is None:
        cv.putText(img, "⏳ CALIBRATING... Show your face", (20, y_pos), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        y_pos += 30
        if face_results.multi_face_landmarks:
            cv.putText(img, "✓ Face detected", (20, y_pos), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv.putText(img, "✓ CALIBRATION COMPLETE", (20, y_pos), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y_pos += 40
        
        # Show height if available
        if current_height > 0:
            # Height in large text
            cv.putText(img, "YOUR HEIGHT:", (20, y_pos), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            y_pos += 50
            
            # Centimeters - LARGE
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
            
            # Measurements count
            cv.putText(img, f"Readings: {len(measurements)}/10", (20, y_pos), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y_pos += 25
            
            # Announcement status
            if height_announced:
                cv.putText(img, "✓ Height announced", (20, y_pos), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            else:
                cv.putText(img, f"⏳ Stabilizing... {len(measurements)}/8", (20, y_pos), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
        else:
            cv.putText(img, "🟡 STEP BACK - SHOW FULL BODY", (20, y_pos), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            y_pos += 35
            cv.putText(img, "Make sure feet are visible", (20, y_pos), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Debug info
    y_pos = h - 120
    cv.putText(img, "DEBUG INFO:", (20, y_pos), 
              cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    y_pos += 20
    for debug in debug_info[-5:]:  # Show last 5 debug messages
        cv.putText(img, debug, (20, y_pos), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y_pos += 15
    
    # Detection status
    if pose_results.pose_landmarks:
        if 'bottom_points' in locals() and bottom_points:
            cv.putText(img, "✓ FULL BODY DETECTED", (20, h-30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv.putText(img, "⚠️ FEET NOT VISIBLE - step back", (20, h-30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv.putText(img, "⚠️ NO BODY DETECTED", (20, h-30), 
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