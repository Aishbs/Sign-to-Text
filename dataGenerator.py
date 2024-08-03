import numpy as np
import progressbar
import cv2
import os
import matplotlib.image as img

class DataGenerator(object):

    def __init__(self):
        super().__init__()

    # Resize frames
    def resize_image(self, image):
        image = img.imread(image)
        image = cv2.resize(image, (64, 64))  # Resize to the desired dimensions
        return image

    def preprocess_image(self, img):
        img = self.resize_image(img)
        return img

    def load_data(self, data_repo, sequence_length=100):
        X = []
        y = []

        labels = os.listdir(data_repo)
        labels.sort()

        for ys, label in enumerate(labels):
            label_path = os.path.join(data_repo, label)
            sub_labels = os.listdir(label_path)

            widgets = ['{}:'.format(label), progressbar.Bar(), progressbar.Percentage(), ' ', '', ' ',
                       progressbar.ETA(), ' ', ' ']
            pbar = progressbar.ProgressBar(widgets=widgets, maxval=len(sub_labels))
            pbar.start()

            for sub_label in sub_labels:
                images_path = os.path.join(label_path, sub_label)
                images = os.listdir(images_path)

                frames = []
                ind = 0
                for img_file in images:
                    img_path = os.path.join(images_path, img_file)
                    x = self.preprocess_image(img_path)
                    frames.append(x)

                # Ensure all sequences have the same length
                if frames:
                    if len(frames) < sequence_length:
                        frames.extend([np.zeros_like(frames[0])] * (sequence_length - len(frames)))
                    elif len(frames) > sequence_length:
                        frames = frames[:sequence_length]

                    # Convert list of frames to a numpy array
                    X.append(np.array(frames))
                else:
                    # Handle the case where frames might be empty
                    X.append(np.zeros((sequence_length, 64, 64, 3)))  # Assuming images are 64x64x3
                y.append(ys)

                # Update progress bar with each processed sub_label
                pbar.update(ind)
                ind += 1

            pbar.finish()

        X = np.array(X)
        y = np.array(y)

        print("Final output shape:", X.shape)
        print("Labels shape:", y.shape)
        return X, y
