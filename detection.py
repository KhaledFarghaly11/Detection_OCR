from fastapi import FastAPI, File, UploadFile

from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.builders import model_builder

import cv2
import numpy as np
import easyocr
import os
import tensorflow as tf

app = FastAPI()
db = []

# Load the model and label map at the start
configs = config_util.get_configs_from_pipeline_file(os.path.join('model', 'pipeline.config'))
detection_model = model_builder.build(model_config=configs['model'], is_training=False)
category_index = label_map_util.create_category_index_from_labelmap(os.path.join('model', 'label_map.pbtxt'))

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join('model', 'ckpt-11')).expect_partial()

# Initialize EasyOCR reader once
reader = easyocr.Reader(['en'])

# Define TensorFlow function globally
@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections

@app.post("/images/")
async def create_upload_file(file: UploadFile = File(...)):
    try:
        IMAGE_PATH = file.filename

        img = cv2.imread(IMAGE_PATH)
        image_np = np.array(img)

        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        region_threshold = 0.05
        def filter_text(region, ocr_result, region_threshold):
            rectangle_size = region.shape[0] * region.shape[1]

            plate = []
            for result in ocr_result:
                length = np.sum(np.subtract(result[0][1], result[0][0]))
                height = np.sum(np.subtract(result[0][2], result[0][1]))

                if length * height / rectangle_size > region_threshold:
                    plate.append(result[1])
            return plate

        detection_threshold = 0.3
        region_threshold = 0.3
        def ocr_it(image, detections, detection_threshold, region_threshold):
            # Scores, boxes and classes above threshold
            scores = list(filter(lambda x: x > detection_threshold, detections['detection_scores']))
            boxes = detections['detection_boxes'][:len(scores)]
            classes = detections['detection_classes'][:len(scores)]

            # Full image dimensions
            width = image.shape[1]
            height = image.shape[0]

            # Apply ROI filtering and OCR
            for idx, box in enumerate(boxes):
                roi = box * [height, width, height, width]
                region = image[int(roi[0]):int(roi[2]), int(roi[1]):int(roi[3])]
                ocr_result = reader.readtext(region)

                text = filter_text(region, ocr_result, region_threshold)

                return text, region

        result, _ = ocr_it(image_np_with_detections, detections, detection_threshold, region_threshold)

        return result
    except Exception as e:
        return "Image not clear"
