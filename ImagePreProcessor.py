from PIL import Image
import cv2
import numpy as np
from skimage import io, color
from skimage.transform import rotate, hough_line, hough_line_peaks
import math
from pytesseract import pytesseract
from matplotlib import pyplot as plt

class ImagePreProcessor:
    def __init__(self, image_path):
        self.image_path = image_path

    def crop_image(self, left, top, right, bottom):
        img = Image.open(self.image_path)
        img_cropped = img.crop((left, top, right, bottom))
        img_cropped.show()
        img_cropped.save("cropped_image.png")  # You can save the cropped image if needed
    
    def skew_stretch_image(self, skew_factor, stretch_factor, output_path):
        img = Image.open(self.image_path)

        # Get the image size
        width, height = img.size

        # Define the transformation matrix
        matrix = [1, skew_factor, 0, stretch_factor, 1, 0]

        # Apply the affine transformation
        skewed_stretched_image = img.transform((width, height), Image.AFFINE, matrix)

        # Save the result
        skewed_stretched_image.show()
        skewed_stretched_image.save(output_path)
    
    # def detect_and_correct_skew(self):
    #    # Open the image
    #     img = Image.open(self.image_path)

    #     # Convert the image to grayscale
    #     gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

    #     # Use pytesseract to get text orientation (skew)
    #     osd_result = pytesseract.image_to_osd(gray)
    #     print("OSD Result:", osd_result)

    #     # Extract the rotate value from the result
    #     skew_angle = float(osd_result.split('\n')[2].split(":")[1])

    #     # Correct the skew
    #     rotated_img = img.rotate(-skew_angle)

    #     # Save or display the result
    #     rotated_img.show()
    #     rotated_img.save("skew_corrected_image.png")
    
    # # Not in use
    # def detect_and_correct_skew(self):
    #     # Open the image
    #     img = io.imread(self.image_path)

    #     # Convert the image to grayscale
    #     gray = color.rgb2gray(img)

    #     # Use scikit-image for automatic skew detection and correction
    #     h, theta, d = hough_line(gray)

    #     # Find the angle with the most prominent line in the Hough space
    #     angle = np.rad2deg(theta[np.argmax(h)])

    #     # Correct the skew
    #     rotated_img = Image.fromarray(rotate(img, angle, mode='constant', cval=1))

    #     # Save or display the result
    #     rotated_img.show()
    #     rotated_img.save("skew_corrected_image.png")

    def automatic_skew_correction(self):    
        cropped_image_path = "cropped_image.png"
        image = cv2.imread(cropped_image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(image)
        plt.title('Original Image')
        plt.show()
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        kernel = np.ones((5, 5), np.float32) / 15
        filtered = cv2.filter2D(gray, -1, kernel)
        plt.imshow(cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB))
        plt.title('Filtered Image')
        plt.show()
        ret, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_OTSU)
        plt.imshow(cv2.cvtColor(thresh, cv2.COLOR_BGR2RGB))
        plt.title('After applying OTSU threshold')
        plt.show()

        adaptive_thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Determine the shape of the input image
        image_shape = image.shape

        # Create an empty canvas with the same shape as the input image
        canvas = np.zeros(image_shape, np.uint8)
        canvas = np.zeros(image_shape, np.uint8)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        cv2.drawContours(canvas, cnt, -1, (0, 255, 255), 3)
        plt.title('Largest Contour')
        plt.imshow(canvas)
        plt.show()
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx_corners = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(canvas, approx_corners, -1, (255, 255, 0), 10)
        approx_corners = sorted(np.concatenate(approx_corners).tolist())
        print('\nThe corner points are ...\n')
        for index, c in enumerate(approx_corners):
            character = chr(65 + index)
            print(character, ':', c)
            cv2.putText(canvas, character, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        # Rearranging the order of the corner points
        approx_corners = [approx_corners[i] for i in [0, 2, 1, 3]]

        plt.imshow(canvas)
        plt.title('Corner Points')
        plt.show()
        w1 = np.sqrt((approx_corners[0][0] - approx_corners[1][0]) ** 2 + (approx_corners[0][1] - approx_corners[1][1]) ** 2)
        w2 = np.sqrt((approx_corners[2][0] - approx_corners[3][0]) ** 2 + (approx_corners[2][1] - approx_corners[3][1]) ** 2)
        w = max(int(w1), int(w2))

        h1 = np.sqrt((approx_corners[0][0] - approx_corners[2][0]) ** 2 + (approx_corners[0][1] - approx_corners[2][1]) ** 2)
        h2 = np.sqrt((approx_corners[1][0] - approx_corners[3][0]) ** 2 + (approx_corners[1][1] - approx_corners[3][1]) ** 2)
        h = max(int(h1), int(h2))

        destination_corners = np.float32([(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)])

        print('\nThe destination points are: \n')
        for index, c in enumerate(destination_corners):
            character = chr(65 + index) + "'"
            print(character, ':', c)
            
        # Define the source points for the homography
        src = np.float32(approx_corners)

        # Define the destination points for the homography
        dst = np.float32(destination_corners)
        print('\nThe approximated height and width of the original image is: \n', (h, w))
        h, w = image.shape[:2]
        H, _ = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        print('\nThe homography matrix is: \n', H)
        un_warped = cv2.warpPerspective(image, H, (w, h), flags=cv2.INTER_LINEAR)


        # plot

        f, (ax1, ax2) = plt.subplots(1, 2)
        ax1.imshow(image)
        x = [src[0][0], src[2][0], src[3][0], src[1][0], src[0][0]]
        y = [src[0][1], src[2][1], src[3][1], src[1][1], src[0][1]]
        ax1.plot(x, y, color='yellow', linewidth=3)
        ax1.set_ylim([h, 0])
        ax1.set_xlim([0, w])
        ax1.set_title('Targeted Area in Original Image')
        ax2.imshow(un_warped)
        ax2.set_title('Unwarped Image')
        plt.show()
        cv2.imwrite("unwarped_image.png", un_warped)