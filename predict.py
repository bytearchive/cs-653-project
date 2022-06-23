import model_code
import utils
from config import Config
import numpy
from sklearn import metrics
import numpy as np
import os

# get the model architectute
deep_model = model_code.get_model()

# load the saved model weights
utils.load_model("./data/model/lastweights.h5", deep_model)

# get the train and test data split
train_datagen, test_datagen = utils.get_test_data()

# Predict with 20% test data
utils.predict(deep_model, test_datagen)

# Predict single images in the `data/test` folder
print("Predicitn Single Images")
from PIL import Image
import numpy as np
from skimage import transform


def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype("float32") / 255
    np_image = transform.resize(np_image, (100, 100, 3))
    np_image = np.expand_dims(np_image, axis=0)
    return np_image


# label -> index map
label_map = test_datagen.class_indices
# inverse index -> label map
inverse_map = {v: k for k, v in label_map.items()}

import os

for i in os.listdir("./data/test/"):
    if i.split(".")[-1] in ["png", "jpg", "jpeg"]:
        print(f"loading ./data/test/{i}")

        img = load(f"./data/test/{i}")
        prediction = deep_model.predict(img)
        y_pred = np.array([np.argmax(x) for x in prediction])
        print(y_pred)
        predicted = inverse_map[y_pred[0]]
        print(f"predicted :{predicted}")
