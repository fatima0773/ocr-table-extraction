from matplotlib import pyplot as plt
import numpy as np
import ImagePreProcessor as preprocessor
import OcrToTableTool as ottt
import TableExtractor as te
import TableLinesRemover as tlr
import cv2

from flask import Flask
from keras import backend as K
from keras.models import load_model
import keras.utils as image
from keras.optimizers import Adam
from imageio import imread
from matplotlib import pyplot as plt
import numpy as np
import ImagePreProcessor as preprocessor
import OcrToTableTool as ottt
import TableExtractor as te
import TableLinesRemover as tlr
import cv2

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_output_decoder import decode_detections

# Initial image dimensions
img_height = 300
img_width = 300

# Set model path
model_path = 'ssd300_pascal_07+12_epoch-62_loss-4.1169_val_loss-4.1424.h5'

# We need to create an SSDLoss object in order to pass that to the model loader.
ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

# Clear previous models from memory.
K.clear_session() 

# Load model
model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                            'L2Normalization': L2Normalization,
                                            'DecodeDetections': DecodeDetections,
                                            'compute_loss': ssd_loss.compute_loss})
orig_images = [] # Store the images here.
input_images = [] # Store resized versions of the images here.

# Set image path
# img_path = './image/selected_1.jpeg'
img_path = './image/selected_2.jpg'
# img_path = './image/selected_3.png'
# img_path = './image/selected_4.jpeg'
# img_path = './image/selected_5.jpeg'
# img_path = './image/selected_6.jpeg'

orig_images.append(imread(img_path))
img = image.load_img(img_path, target_size=(img_height, img_width))
img = image.img_to_array(img) 
input_images.append(img)
input_images = np.array(input_images)
y_pred = model.predict(input_images)

# Print results
print(y_pred.shape)
normalize_coords = True
y_pred_thresh = decode_detections(y_pred,
                                confidence_thresh=0.1,
                                iou_threshold=0.4,
                                top_k=1,
                                normalize_coords=normalize_coords,
                                img_height=img_height,
                                img_width=img_width)
np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print('class, conf, xmin, ymin, xmax, ymax')
print(y_pred_thresh[0])
for box in y_pred_thresh[0]:
    print(box)
    # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
    xmin = box[2] * orig_images[0].shape[1] / img_width
    ymin = box[3] * orig_images[0].shape[0] / img_height
    xmax = box[4] * orig_images[0].shape[1] / img_width
    ymax = box[5] * orig_images[0].shape[0] / img_height
preprocessor = preprocessor.ImagePreProcessor(img_path)
preprocessor.crop_image(left=xmin, top=ymin, right=xmax, bottom=ymax)
# preprocessor.automatic_skew_correction()
# table_extractor = te.TableExtractor("unwarped_image.jpg")
table_extractor = te.TableExtractor(img_path)
perspective_corrected_image = table_extractor.execute()
cv2.imshow("perspective_corrected_image", perspective_corrected_image)
lines_remover = tlr.TableLinesRemover(perspective_corrected_image)
image_without_lines = lines_remover.execute()
cv2.imshow("image_without_lines", image_without_lines)

ocr_tool = ottt.OcrToTableTool(image_without_lines, perspective_corrected_image)
ocr_tool.execute()

cv2.waitKey(0)
cv2.destroyAllWindows()




