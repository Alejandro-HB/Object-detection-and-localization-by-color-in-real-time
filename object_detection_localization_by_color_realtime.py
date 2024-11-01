
import pyrealsense2 as rs
import cv2
import numpy as np

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def preprocess_image(frame):
    # Load and convert image to HSV
    input_image = frame
    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)
    return input_image, hsv_image

def apply_threshold(hsv_image):
    # Extract and blur S channel for mask
    S = hsv_image[:, :, 1]
    blurred_image = cv2.medianBlur(S, 15)
    cv2.imshow('blurred image', blurred_image)
    _, mask = cv2.threshold(blurred_image, 111, 255, cv2.THRESH_BINARY)
    return mask

def segment_with_kmeans(segmented_image, K=4):
    # Prepare image for k-means
    vectorized = segmented_image.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(vectorized, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape(segmented_image.shape)
    return result_image


def draw_contours_on_image(input_image, depth_image, mask, model):
    # Convert to grayscale for contour detection
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours, bounding boxes, and centroids
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w<30 or h<30: continue
        cv2.rectangle(input_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Centroid
        cx=x + w // 2
        cy=y + h // 2
        cv2.circle(input_image, (cx, cy), 2, (0, 0, 255), -1)
        # Distance to camera
        distance=depth_image[cy, cx]
        # Formated distance
        f_distance=format(distance/1000,'.2f')
        cv2.putText(input_image, str(f_distance+'m'), (cx-10, cy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1, cv2.LINE_AA)
        # Object dimensions
        pixel_size=model.predict(pd.DataFrame({'distance':[distance]}))
        width=w*pixel_size
        height=h*pixel_size
        # Width
        # Formated width
        f_width=format(width[0]/1000, '.2f')
        cv2.putText(input_image, str(f_width+'m'), (x + w // 2 - 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1, cv2.LINE_AA)
        # Height
        # Formated height
        f_height=format(height[0]/1000, '.2f')
        cv2.putText(input_image, str(f_height+'m'), (x - w // 2, y + h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1, cv2.LINE_AA)
    
    return input_image



# Create a pipeline
pipeline = rs.pipeline()

# Create a config and configure the pipeline to stream
#  different resolutions of color and depth streams
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
profile = pipeline.start(config)

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 2 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Lineal regression model
# Step 1: Load or create your data
# Replace with your actual data
data = {
    'distance':   [600, 1200, 1800, 2400, 3000, 3600],  # Distances from the camera
    'pixel_size': [0.98, 1.98, 2.91, 3.97, 4.93, 5.95]  # Pixel sizes observed in the image
}
df = pd.DataFrame(data)

# Separate features and target variable
X = df[['distance']]  # Only distance is the feature
y = df['pixel_size']

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 3: Create and train the simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)


i=0
# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        colorr_image=color_image.copy()

        # Remove background - Set pixels further than clipping_distance to grey
        grey_color = 153
        depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

        # Render images:
        #   depth align to color on left
        #   depth on right
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        images = np.hstack((color_image, depth_colormap))
        
        #image segmentation
        # bb_image=segmentation_BB(color_image)
        input_image, hsv_image = preprocess_image(color_image)
        
        mask = apply_threshold(hsv_image)

        # Create segmented image using mask
        mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        segmented_image = cv2.bitwise_and(input_image, mask_3)

        # Apply k-means clustering
        kmeans_result = segment_with_kmeans(segmented_image)

        # Display images
        # cv2.imshow('Saturation Channel', hsv_image[:, :, 1])
        # cv2.imshow('Mask', mask)
        # cv2.imshow('Segmented Image', segmented_image)
        # cv2.imshow('K-means Segmented Image', kmeans_result)

        # Draw final contours
        bounding_box_image=draw_contours_on_image(input_image, depth_image, kmeans_result, model)
        
        # cv2.namedWindow('Depth image', cv2.WINDOW_NORMAL)
        # cv2.imshow('Depth image', depth_image_norm)
        # cv2.namedWindow('Segmentation', cv2.WINDOW_NORMAL)
        # cv2.imshow('Segmentation', kmeans_result)
        cv2.namedWindow('Bounding boxes', cv2.WINDOW_NORMAL)
        cv2.imshow('Bounding boxes', bounding_box_image)
        # cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        # cv2.imshow('Align Example', images)
        key = cv2.waitKey(1)
        
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.imwrite('./images/colormap.png',depth_colormap)
            cv2.imwrite('./images/colorImage.png',colorr_image)
            cv2.imwrite('./images/depthImage.png',depth_image)
            cv2.imwrite('./images/finalResult.png', color_image)
            cv2.destroyAllWindows()
            break
        elif key==13:
            cv2.imwrite('./images/test'+str(i)+'.png', colorr_image)
            i+=1
            
finally:
    pipeline.stop()
