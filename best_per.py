import cv2
import numpy as np

def process_frame(frame):
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range for yellow color in HSV space
    lower_yellow = np.array([15, 100, 100])  # Adjust based on your requirement
    upper_yellow = np.array([35, 255, 255])  # Adjust based on your requirement

    # Create a mask that only allows yellow pixels to pass
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Invert the yellow mask to get non-yellow areas
    non_yellow_mask = cv2.bitwise_not(yellow_mask)

    # Apply the non-yellow mask to the original image
    non_yellow_image = cv2.bitwise_and(frame, frame, mask=non_yellow_mask)

    # Define blob detector parameters
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 50
    params.maxArea = 5000
    params.filterByCircularity = True
    params.minCircularity = 0.8
    params.filterByConvexity = True
    params.minConvexity = 0.9
    params.filterByInertia = True
    params.minInertiaRatio = 0.9

    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs in the non-yellow image
    keypoints = detector.detect(non_yellow_image)

    # Draw keypoints on the original frame
    img_with_blobs = draw_thicker_blobs(frame, keypoints, color=(0, 255, 255), thickness=3)

    # Display the number of detected dots
    num_dots = len(keypoints)
    
    # Check if no dots are detected
    if num_dots == 0:
        max_dots[0] = 0
    else:
        # Update the maximum number of dots detected
        if num_dots > max_dots[0]:
            max_dots[0] = num_dots

    # Update the text to display the current and maximum number of dots
    img_with_txt = add_text(img_with_blobs, f'Dots: {num_dots} (Max: {max_dots[0]})')

    return img_with_txt

def draw_thicker_blobs(image, keypoints, color=(0, 255, 255), thickness=2):
    # Draw thicker circles around the detected keypoints
    image_with_blobs = image.copy()
    for kp in keypoints:
        center = (int(kp.pt[0]), int(kp.pt[1]))
        radius = int(kp.size / 2)
        cv2.circle(image_with_blobs, center, radius, color, thickness)
    return image_with_blobs

def add_text(img, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (20, 40)
    fontScale = 1
    fontColor = (0, 0, 0)
    lineType = 2

    img_with_txt = cv2.putText(img, text, bottomLeftCornerOfText, font, fontScale, fontColor, lineType)
    return img_with_txt

def main():
    global max_dots
    max_dots = [0]  # Initialize a list to keep track of the maximum number of dots

    # Open the camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Failed to grab frame")
            break

        # Process the current frame
        processed_frame = process_frame(frame)

        # Display the processed frame
        cv2.imshow('Live Dot Recognition', processed_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
