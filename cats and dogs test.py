from PIL import Image
import tensorflow as tf
import cv2
import os

DATASET_PATH = 'dataset/'
num_classes = len(os.listdir(DATASET_PATH))
class_mode = "binary" if num_classes == 2 else "categorical"

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"error: file not found at path: {image_path}")
        return
    try:
        img = Image.open(image_path)
        img.verify()
        img = Image.open(image_path)
    except (OSError, IOError):
        print(f"error: corrupted image - {image_path}")
        return
    model = tf.keras.models.load_model("image_classifier.keras")
    img = cv2.imread(image_path)

    if img is None:
        print(f"error:failed to read image - {image_path}")
        return
    img = cv2.resize(img, (128, 128))
    img = img / 255
    img = tf.expand_dims(img, axis=0)

    prediction = model.predict(img)

    class_names = os.listdir(DATASET_PATH)

    if class_mode == "binary":
        predicted_class = class_names[int(bool(prediction[0] > 0.5))]
    else:
        predicted_class = class_names[tf.argmax(prediction,axis=-1).numpy()[0]]

    print(f"model determined: {predicted_class}")

predict_image("cat1.jpg")