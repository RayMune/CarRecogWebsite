from flask import Flask, request, render_template
import cv2
import numpy as np

app = Flask(__name__)

def analyze_image(image):
    """
    Analyzes an image to find the dominant color of a car and detect its number plate.

    Args:
        image: The input image as a NumPy array.

    Returns:
        A dictionary containing the analysis results:
            - 'color': The dominant color of the car ('Red', 'Blue', 'Green') or None if not found.
            - 'number_plates': A list of detected number plate regions (bounding boxes) represented as dictionaries
                               with keys 'x', 'y', 'w', and 'h', or an empty list.
    """

    # Preprocess the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise

    # Dominant color analysis (heuristic approach)
    # Convert to HSV for easier color separation
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges (adjust these based on your needs)
    red_lower = (170, 100, 100)
    red_upper = (180, 255, 255)
    blue_lower = (100, 100, 100)
    blue_upper = (140, 255, 255)
    green_lower = (40, 50, 50)
    green_upper = (80, 255, 255)

    # Count pixels within each color range
    red_count = cv2.countNonZero(cv2.inRange(hsv, red_lower, red_upper))
    blue_count = cv2.countNonZero(cv2.inRange(hsv, blue_lower, blue_upper))
    green_count = cv2.countNonZero(cv2.inRange(hsv, green_lower, green_upper))

    # Find the dominant color based on pixel counts
    dominant_color = None
    if red_count > blue_count and red_count > green_count:
        dominant_color = 'Red'
    elif blue_count > red_count and blue_count > green_count:
        dominant_color = 'Blue'
    elif green_count > red_count and green_count > blue_count:
        dominant_color = 'Green'

    # Number plate detection (basic approach)
    # Apply Canny edge detection
    edges = cv2.Canny(blur, 200, 400)

    # Find contours
    cnts = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # Filter contours based on aspect ratio and area
    number_plates = []
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w) / h
        if 2 < aspect_ratio < 5 and 100 < cv2.contourArea(cnt) < 5000:  # Adjust thresholds as needed
            number_plates.append({'x': x, 'y': y, 'w': w, 'h': h})

    return {'color': dominant_color, 'number_plates': number_plates}

def allowed_file(filename):
    """
    Checks if the uploaded file has a valid image extension.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        print(request.files)
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']

        # Check if the file is empty
        if file.filename == '':
            return render_template('index.html', error='No selected file')

        # Check if the file is an image
        if file and allowed_file(file.filename):
            # Read the uploaded image
            image_stream = file.read()
            nparr = np.fromstring(image_stream, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Analyze the image
            analysis_result = analyze_image(image)

            # Display the result to the user
            return render_template('result.html', result=analysis_result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
    