from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PyQt5.uic import loadUi
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('malaria_detection_model_final.h5')

class_names = {0: 'Parasite', 1: 'Uninfected'}

class App(QMainWindow):
    def __init__(self):
        super().__init__()
        loadUi("designer.ui", self)

        # Connect signals and slots
        self.select_button.clicked.connect(self.open_file)
        self.detect_button.clicked.connect(self.detect_disease)

        # Initialize variable to store image array
        self.img_array = None

    def open_file(self):
        self.file_path, _ = QFileDialog.getOpenFileName(self, 'Open Image File', '', 'Image Files (*.png *.jpg *.jpeg *.bmp *.gif)')
        if self.file_path:
            img = Image.open(self.file_path)

            # Get canvas dimensions
            canvas_width = self.canvas.width()
            canvas_height = self.canvas.height()

            # Calculate aspect ratio
            img_aspect_ratio = img.width / img.height
            canvas_aspect_ratio = canvas_width / canvas_height

            # Resize image to fit the canvas while maintaining aspect ratio
            if img_aspect_ratio > canvas_aspect_ratio:
                new_width = canvas_width
                new_height = int(canvas_width / img_aspect_ratio)
            else:
                new_width = int(canvas_height * img_aspect_ratio)
                new_height = canvas_height

            img = img.resize((new_width, new_height), Image.LANCZOS)
            self.img_array = np.array(img)

            # Converts PIL Image to QImage
            height, width, channel = self.img_array.shape
            bytes_per_line = 3 * width
            q_image = QImage(self.img_array.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.canvas.setPixmap(pixmap)
            self.detect_button.setEnabled(True)

    def detect_disease(self):
        if self.img_array is not None:
            try:
                image = Image.fromarray(self.img_array)
                image = image.resize((224, 224))
                image = np.array(image) / 255.0  # Normalize the image
                image = np.expand_dims(image, axis=0)  # Add batch dimension

                # Make predictions
                predictions = model.predict(image)

                # Update result label
                predicted_class_index = np.argmax(predictions)
                predicted_class = class_names.get(predicted_class_index, 'Unknown')
                result_text = f"Predicted Class: {predicted_class}"
                self.result_label.setText(result_text)

            except Exception as e:
                QMessageBox.warning(self, "Error", "Error occurred during detection.")
        else:
            QMessageBox.warning(self, "No Image", "Please upload an image first.")

if __name__ == "__main__":
    app = QApplication([])
    root = App()
    root.show()
    app.exec_()
