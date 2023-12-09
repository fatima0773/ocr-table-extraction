from PIL import Image
import cv2
import numpy as np
from skimage import io, color
from skimage.transform import rotate, hough_line, hough_line_peaks
import math
from pytesseract import pytesseract

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
    
    def detect_and_correct_skew(self):
       # Open the image
        img = Image.open(self.image_path)

        # Convert the image to grayscale
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

        # Use pytesseract to get text orientation (skew)
        osd_result = pytesseract.image_to_osd(gray)
        print("OSD Result:", osd_result)

        # Extract the rotate value from the result
        skew_angle = float(osd_result.split('\n')[2].split(":")[1])

        # Correct the skew
        rotated_img = img.rotate(-skew_angle)

        # Save or display the result
        rotated_img.show()
        rotated_img.save("skew_corrected_image.png")
    
    def detect_and_correct_skew(self):
        # Open the image
        img = io.imread(self.image_path)

        # Convert the image to grayscale
        gray = color.rgb2gray(img)

        # Use scikit-image for automatic skew detection and correction
        h, theta, d = hough_line(gray)

        # Find the angle with the most prominent line in the Hough space
        angle = np.rad2deg(theta[np.argmax(h)])

        # Correct the skew
        rotated_img = Image.fromarray(rotate(img, angle, mode='constant', cval=1))

        # Save or display the result
        rotated_img.show()
        rotated_img.save("skew_corrected_image.png")

    # def detect_and_correct_skew(self):
    #     # Open the image
    #     img = cv2.imread(self.image_path)

    #     # Convert the image to grayscale
    #     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #     # Use Canny edge detection to find edges in the image
    #     edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    #     # Use Hough Line Transform to detect lines in the image
    #     lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    #     # Calculate the average angle of detected lines
    #     angles = []
    #     for line in lines:
    #         for rho, theta in line:
    #             angles.append(theta)

    #     average_angle = np.mean(angles)

    #     # Correct the skew
    #     rotated_img = Image.fromarray(self.rotate_image(img, average_angle))

    #     # Save or display the result
    #     rotated_img.show()
    #     rotated_img.save("skew_corrected_image.png")

    # @staticmethod
    # def rotate_image(image, angle):
    #     # Get image center and rotation matrix
    #     center = tuple(np.array(image.shape[1::-1]) / 2)
    #     rot_mat = cv2.getRotationMatrix2D(center, -np.degrees(angle), 1.0)  # Adjust the rotation direction

    #     # Perform the affine transformation
    #     rotated_image = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

    #     return rotated_image