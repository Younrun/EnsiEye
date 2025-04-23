# Known distances and object widths in centimeters
KNOWN_DISTANCE = 45  # cm
PERSON_WIDTH = 20    # cm
CAR_WIDTH = 50       # cm

# Class names for detection
CLASS_NAMES = ['Car', 'Cyclist', 'Motorcycle', 'Pedestrian', 'Truck']
SIGN_CLASS_NAMES = ['speedlimit', 'crosswalk', 'stop', 'trafficlight']

# Reference image paths
REF_IMAGES_PATHS = {
    'Pedestrian': 'assets/image_9.jpg',
    'Car': 'assets/traficsign_1.jpg'
}

# Color definitions (BGR format)
COLORS = {
    'green': (0, 255, 0),
    'red': (0, 0, 255),
    'yellow': (0, 255, 255),
    'black': (0, 0, 0)
}

# Font
FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
