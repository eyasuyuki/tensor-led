import unittest

import cv2
from keras.engine.saving import model_from_json
from keras_preprocessing.image import load_img, img_to_array

test_files=["0/n0.png",
            "1/n1.png",
            "2/n2.png",
            "3/n3.png",
            "4/n4.png",
            "5/n5.png",
            "6/n6.png",
            "6/n6b.png",
            "7/n7.png",
            "7/n7b.png",
            "8/n8.png",
            "8/n9.png"]


class MyTestCase(unittest.TestCase):
    def test_something(self):
        # read trained keras model

        f = open("./train.json")
        json_text = f.read()
        model = model_from_json(json_text)
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.5), metrics=['accuracy'])
        model.load_weights("./train.hdf5")

        test = []
        for file_name in test_files:
            tmp_img = load_img(f"./images/{file_name}", grayscale=True, target_size=(64, 64))
            tmp_img_array = img_to_array(tmp_img)
            tmp_img_array /= 255
            test.append(tmp_img_array)

        result = model.predict_classes(test)
        self.assertEqual(result, "012345667789")
        

if __name__ == '__main__':
    unittest.main()
