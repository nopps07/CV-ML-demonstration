# libraries
import cv2
# from cv2 import cv2
import numpy as np

# print(cv2.__version__)

path = "C:/Users/gunho/PycharmProjects/CV-ML-demonstration/data/images/gunho.png"
# image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
image = cv2.imread(path, cv2.IMREAD_COLOR)

print(image.shape)

# values close to 0: darker pixels
# values close to 255: brighter pixels
print(image)
print(np.amax(image))
print(np.amin(image))

cv2.imshow('Computer Vision', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

## IMAGE PROCESSING
path = "C:/Users/gunho/PycharmProjects/CV-ML-demonstration/data/images/gunho.png"
original_image = cv2.imread(path, cv2.IMREAD_COLOR)

# trasnform the image into grayscale
# opencv handles BGR instead of RGB
gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

# Blur
kernel = np.ones((5, 5)) / 25  # normalize
# Laplacian (edge)
kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
# sharp
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
print(kernel)

# -1 "destination depth"
blur_image = cv2.filter2D(original_image, -1, kernel)
edge_image = cv2.filter2D(original_image, -1, kernel)
edge_image = cv2.Laplacian(original_image, -1)  # same
sharp_image = cv2.filter2D(original_image, -1, kernel)

# gaussian blur is used to reduce noise!

cv2.imshow('Original Image', original_image)
cv2.imshow('Original Image', sharp_image)
# cv2.imshow('Blurred Image', blur_image)
# cv2.imshow('Gray Image', gray_image)
# cv2.imshow('Edge Image', edge_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


## Lane detection
import cv2
import numpy as np

def draw_the_lines(image, lines):
    # create a distinct image for the lines [0,255] - all 0 values means black image
    lines_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # there are (x,y) for the starting and end points of the lines
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(lines_image, (x1, y1), (x2, y2), (255, 0, 0), thickness=3)

    # finally we have to merge the image with the lines
    image_with_lines = cv2.addWeighted(image, 0.8, lines_image, 1, 0.0)

    return image_with_lines


def region_of_interest(image, region_points):
    # replace pixels with 0 (black) for the regions we are not interested
    mask = np.zeros_like(image)

    # interested region -> the lower triangle - 255 white pixels
    cv2.fillPoly(mask, region_points, 255)

    # use the mask!
    # logical and operator
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image


def get_detected_lanes(image):
    (height, width) = (image.shape[0], image.shape[1])

    # turn the image into grayscale
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # edge detection kernel (Canny's algorithm)
    canny_image = cv2.Canny(gray_image, 100, 120)  # 100 = low threshold, 120 = high threshold

    # interest region -> "lower region" = driving lanes. Discard the other regions
    region_of_interest_vertices = [
        (0, height),
        (width / 2, height * 0.65),
        (width, height)
    ]

    # get rid of the irrelevant parts
    cropped_image = region_of_interest(canny_image, np.array([region_of_interest_vertices], np.int32))

    # Use the line detection algorithm (radians instead of degrees! 1 degree = pi / 180)
    lines = cv2.HoughLinesP(cropped_image, rho=2, theta=np.pi / 180, threshold=50, lines=np.array([]),
                            minLineLength=40, maxLineGap=150)

    # draw the lines on the image
    image_with_lines = draw_the_lines(image, lines)

    return image_with_lines


path = "C:/Users/gunho/PycharmProjects/CV-ML-demonstration/data/videos/lane_detection_video.mp4"
video = cv2.VideoCapture(path)

while video.isOpened():

    is_grabbed, frame = video.read()

    # due to the end of the video
    if not is_grabbed:
        break

    frame = get_detected_lanes(frame)

    cv2.imshow('Lane Detection Video', frame)  # very fast
    cv2.waitKey(20)  # delay the application #the larger the slower

video.release()
cv2.destroyAllWindows()
