import cv2 as cv
from ultralytics import YOLO
import numpy as np

class VirtualCamera:
    """
    Professional virtual camera system for cinematic auto-reframing.
    Mimics human camera operator behavior with smooth motion and composition awareness.
    """
    
    def __init__(self, frame_width, frame_height, output_width=1080, output_height=1920):
        self.frame_w = frame_width
        self.frame_h = frame_height
        self.output_w = output_width
        self.output_h = output_height
        
        # Fixed crop width for stable framing (9:16 aspect ratio)
        self.crop_h = frame_height
        self.crop_w = int(frame_height * 9 / 16)
        
        # Virtual camera position (center of crop)
        self.camera_x = frame_width // 2
        
        # Smoothing parameters for natural motion
        self.smoothing_factor = 0.85  # Higher = slower, more cinematic movement
        self.dead_zone = 30  # Pixels: camera won't move for tiny subject shifts
        self.max_velocity = 25  # Max pixels per frame to limit sudden jumps
        
        # Composition margins (safe area around subject)
        self.face_margin_left = 80
        self.face_margin_right = 80
        self.face_margin_top = 120
        self.face_margin_bottom = 120
        
        # Ensure crop fits in frame
        if self.crop_w > frame_width:
            self.crop_w = frame_width
    
    def get_safe_crop_bounds(self, subject_center_x, subject_left, subject_right):
        """
        Calculate safe crop bounds that keep subject visible with margins.
        Prevents face from touching frame edges.
        """
        # Desired framing: place subject with left margin
        desired_center = subject_center_x
        
        # Calculate safe region where subject won't touch edges
        min_crop_center = self.face_margin_left + (self.crop_w // 2)
        max_crop_center = self.frame_w - self.face_margin_right - (self.crop_w // 2)
        
        # Clamp desired center to safe region
        safe_center = np.clip(desired_center, min_crop_center, max_crop_center)
        
        return safe_center
    
    def apply_dead_zone(self, target_x, current_x):
        """
        Dead zone logic: camera doesn't move for tiny subject movements.
        Reduces jitter and makes framing more stable.
        """
        delta = abs(target_x - current_x)
        
        if delta < self.dead_zone:
            return current_x
        
        return target_x
    
    def apply_smoothing(self, target_x):
        """
        Exponential smoothing for natural camera motion.
        Mimics human camera operator inertia.
        """
        # Calculate velocity
        delta = target_x - self.camera_x
        
        # Limit velocity to prevent sudden jumps
        delta = np.clip(delta, -self.max_velocity, self.max_velocity)
        
        # Apply smoothing
        self.camera_x = int(self.camera_x + delta * (1.0 - self.smoothing_factor))
        
        return self.camera_x
    
    def get_crop_region(self, subject_center_x, subject_left, subject_right):
        """
        Get final crop region with all constraints applied.
        Returns (left, right, top, bottom)
        """
        # Get safe bounds considering margins
        safe_center = self.get_safe_crop_bounds(subject_center_x, subject_left, subject_right)
        
        # Apply dead zone to reduce jitter
        target_x = self.apply_dead_zone(safe_center, self.camera_x)
        
        # Apply smoothing for natural motion
        final_x = self.apply_smoothing(target_x)
        
        # Calculate crop boundaries
        left = final_x - self.crop_w // 2
        right = left + self.crop_w
        
        # Hard clamp to frame boundaries
        left = np.clip(left, 0, self.frame_w - self.crop_w)
        right = left + self.crop_w
        
        top = 0
        bottom = self.crop_h
        
        return int(left), int(right), int(top), int(bottom)
    
    def crop_frame(self, frame, subject_center_x, subject_left, subject_right):
        """
        Crop frame to vertical format with professional framing.
        """
        left, right, top, bottom = self.get_crop_region(
            subject_center_x, subject_left, subject_right
        )
        
        cropped = frame[top:bottom, left:right]
        
        return cropped


def get_largest_person(results):
    """
    Extract the largest detected person from YOLO results.
    Returns: (x1, y1, x2, y2, center_x) or None
    """
    largest_area = 0
    best_box = None
    
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            
            # Class 0 = person in COCO dataset
            if cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                
                if area > largest_area:
                    largest_area = area
                    best_box = (x1, y1, x2, y2)
    
    if best_box:
        x1, y1, x2, y2 = best_box
        center_x = (x1 + x2) // 2
        return x1, y1, x2, y2, center_x
    
    return None


def main():
    # Load YOLO model
    model = YOLO("yolov8n.pt")
    
    # Open video
    cap = cv.VideoCapture("demo.mp4")
    
    if not cap.isOpened():
        print("Error: Cannot open video. Please check 'demo.mp4'")
        exit(1)
    
    # Video properties
    fps = int(cap.get(cv.CAP_PROP_FPS))
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {frame_width}x{frame_height} @ {fps} FPS")
    
    # Output settings
    output_width = 1080
    output_height = 1920
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter("output1.mp4", fourcc, fps, (output_width, output_height))
    
    # Initialize virtual camera
    camera = VirtualCamera(frame_width, frame_height, output_width, output_height)
    
    frame_count = 0
    
    print("Processing video... (Press 'q' to quit during playback)")
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame_count += 1
        
        # Run YOLO detection
        results = model(frame)
        
        # Get largest person
        person = get_largest_person(results)
        
        if person:
            x1, y1, x2, y2, center_x = person
            
            # Get professional crop with virtual camera
            cropped = camera.crop_frame(frame, center_x, x1, x2)
            
            # Resize to output format maintaining aspect ratio
            # This prevents stretching
            h, w = cropped.shape[:2]
            
            # If crop doesn't exactly match output dimensions, resize properly
            if w != output_width or h != output_height:
                # Resize maintaining aspect ratio
                scale = min(output_width / w, output_height / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                resized = cv.resize(cropped, (new_w, new_h), interpolation=cv.INTER_LINEAR)
                
                # Create canvas and place resized frame (centered)
                final_output = np.zeros((output_height, output_width, 3), dtype=np.uint8)
                x_offset = (output_width - new_w) // 2
                y_offset = (output_height - new_h) // 2
                final_output[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
            else:
                final_output = cv.resize(cropped, (output_width, output_height))
            
            # Write to output video
            out.write(final_output)
            
            # Display preview
            cv.imshow("Auto Reframe", final_output)
            
            # Allow quit during playback
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            # No person detected, output black frame with center crop
            center_crop_left = max(0, frame_width // 2 - camera.crop_w // 2)
            center_crop_right = min(frame_width, center_crop_left + camera.crop_w)
            center_crop = frame[:, center_crop_left:center_crop_right]
            
            final_output = cv.resize(center_crop, (output_width, output_height))
            out.write(final_output)
            cv.imshow("Auto Reframe", final_output)
            
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
    
    # Cleanup
    cap.release()
    out.release()
    cv.destroyAllWindows()
    
    print(f"✓ Processing complete! Output saved to 'output1.mp4' ({frame_count} frames)")


if __name__ == "__main__":
    main()
