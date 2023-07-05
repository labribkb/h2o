import numpy as np
import os
from PIL import Image
import tensorflow as tf
from h2o import H2O

# Avoids using the entire gpu(s) capacities if not necessary
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for dev in physical_devices:
        tf.config.experimental.set_memory_growth(dev, True)


# The tf.function is used as an empirical optimization for neural networks
@tf.function
def tf_predict(model, inputs):
    return model(inputs, training=False)


class Main_H2O():
    def __init__(self) -> None:     
        # 1) Provide your own model
        model_builder = tf.keras.applications.xception.Xception
        test_model = model_builder(weights="imagenet")
        preprocess_input = tf.keras.applications.xception.preprocess_input
        decode_predictions = tf.keras.applications.xception.decode_predictions
        trg_size = (299, 299)

        # 2) Instantiate H2O explanation method with:
        #       - n: the maximum number of superpixels per segmentation in the hierarchy
        #       - min_size: the number of pixels below which a superpixel is not segmented anymore in the hierarchy
        #       - predict_fct: the function to make a model prediction on an image
        #       - nb_classes: the number of output classes in the classification task
        nb_classes = test_model.layers[-1].output.shape[-1]
        explanable = H2O(n=4, min_size=trg_size[0], predict_fct=tf_predict, nb_classes=nb_classes)

        # 3) Explain the image examples provided in the *input_dir* folder
        input_dir = "./imagenet"
        output_dir = "results_demo"
        for image_file in os.listdir(input_dir):
            # 3.1) Load and preprocess image
            img_path = f"{input_dir}/{image_file}"
            image_name = image_file[:-5] # Remove ".JPEG" extension
            original_image = self.get_img_array(img_path, trg_size)
            preprocessed_img = preprocess_input(np.copy(original_image))

            # 3.2) Feed the preprocessed image to the model (console display)
            preds = test_model.predict(preprocessed_img)
            print("Predicted:", decode_predictions(preds, top=5)[:5])

            saliency = explanable.h2o_explain_image(preprocessed_img, test_model, subdivision_per_channels=3, slic_sigma=2)
            heatmap = explanable.final_heatmap(
                original_image, saliency, apply_gaussian=True, sm_save_path=f"./{output_dir}/{image_name}.npy")
            PIL_image = Image.fromarray(np.squeeze(heatmap))
            PIL_image.save(f"./{output_dir}/{image_name}.jpg", "JPEG")


    def get_img_array(self, image_path, size):
        # `img` is a PIL image of size 299x299
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=size)
        # `array` is a float32 Numpy array of shape (299, 299, 3)
        array = tf.keras.preprocessing.image.img_to_array(img)
        # We add a dimension to transform our array into a "batch"
        # of size (1, 299, 299, 3)
        array = np.expand_dims(array, axis=0)
        return array


Main_H2O()
