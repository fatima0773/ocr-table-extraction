from matplotlib import pyplot as plt
import numpy as np
import ImagePreProcessor as preprocessor
import OcrToTableTool as ottt
import TableExtractor as te
import TableLinesRemover as tlr
import cv2

path_to_image = "./image/table4.jpeg"



preprocessor = preprocessor.ImagePreProcessor(path_to_image)
# 81.6050033569336 265.9497985839844 562.9752502441406 886.4661865234375
# 148.68987464904785 524.782829284668 420.8183898925781 880.7432556152344
xmin = 148.6
ymin = 524.7
xmax = 420.8
ymax = 880.7
preprocessor.crop_image(left=xmin, top=ymin, right=xmax, bottom=ymax)
# preprocessor.detect_and_correct_skew()

image = cv2.imread("./image/test_shot_4.jpeg")
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
# adaptive_thresh = cv2.adaptiveThreshold(filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# plt.imshow(cv2.cvtColor(adaptive_thresh, cv2.COLOR_BGR2RGB))
# plt.title('After applying Adaptive Thresholding')
# plt.show()
ret, thresh = cv2.threshold(filtered, 250, 255, cv2.THRESH_OTSU)
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
# un_warped.save("unwarped_image.png")
# Save the unwarped image
cv2.imwrite("unwarped_image.png", un_warped)

# cropped = un_warped[0:h, 0:w]
# plt.imshow(cropped)
# plt.show()


# 81.6050033569336 265.9497985839844 562.9752502441406 886.4661865234375

# skew_factor = 0.1
# stretch_factor = 0.1
output_image_path = "unwarped_image.png"
output_image = cv2.imread(output_image_path)
cv2.imshow("OUTPUTTTTT", output_image)
# output_image_path.show()
print(output_image_path)
# preprocessor.skew_stretch_image(skew_factor, stretch_factor, output_image_path)
# preprocessor.skew_stretch_image()
table_extractor = te.TableExtractor(output_image_path)
perspective_corrected_image = table_extractor.execute()
cv2.imshow("perspective_corrected_image", perspective_corrected_image)


lines_remover = tlr.TableLinesRemover(perspective_corrected_image)
image_without_lines = lines_remover.execute()
cv2.imshow("image_without_lines", image_without_lines)

ocr_tool = ottt.OcrToTableTool(image_without_lines, perspective_corrected_image)
ocr_tool.execute()

cv2.waitKey(0)
cv2.destroyAllWindows()