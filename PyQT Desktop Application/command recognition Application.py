import sys
import librosa
import tensorflow as tf
from PyQt5.QtWidgets import QApplication, QWidget, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit, QPushButton
from PyQt5.QtCore import Qt, QTimer
import sounddevice as sd
import numpy as np
from PyQt5.QtGui import QPainter, QColor



# sample_rate = sd.query_devices(0)["default_samplerate"]
SAVED_MODEL_PATH = "./model.h5"


class KeywordBox(QLineEdit):
    def __init__(self, keyword, predicted_keyword):
        super().__init__(keyword)
        self.setReadOnly(True)
        self.predicted_keyword = predicted_keyword
    
    def paintEvent(self, event):
        painter = QPainter(self)
        if self.text() == self.predicted_keyword:
            painter.fillRect(event.rect(), QColor(Qt.green))
        else:
            painter.fillRect(event.rect(), QColor(Qt.white))
        super().paintEvent(event)


class App(QWidget):
    def __init__(self):
        super().__init__()
        self.sample_rate = 22050
        self.SAMPLES_TO_CONSIDER = 22050
        self.MFCCs = None
        self.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
        # create 21 random keywords
        # self.keywords = [f"Keyword {i+1}" for i in range(21)]
        # random.shuffle(self.keywords)
        self._mapping = [
        "down",
        "eight",
        "five",
        "four",
        "go",
        "happy",
        "left",
        "nine",
        "no",
        "off",
        "one",
        "right",
        "seven",
        "six",
        "stop",
        "three",
        "tree",
        "two",
        "up",
        "yes",
        "zero"
        ]

        # create text boxes for each keyword
        # self.textboxes = [KeywordBox(keyword) for keyword in self._mapping]
        self.textboxes = [KeywordBox(keyword, "") for keyword in self._mapping]


        # create record and play buttons
        self.setWindowTitle("Command Recognition Application")

        self.record_button = QPushButton("Record")
        self.predicted_button = QPushButton("Predict")
        self.play_button = QPushButton("Play")
        self.predicted_button.setDisabled(True)
        self.play_button.setDisabled(True)


        # set up layout
        hbox = QHBoxLayout()
        vbox = QVBoxLayout()

        


        for i in range(7):
            hbox.addWidget(self.textboxes[i])
            hbox.addWidget(self.textboxes[i+7])
            hbox.addWidget(self.textboxes[i+14])
            vbox.addLayout(hbox)
            hbox = QHBoxLayout()

        vbox.addWidget(self.record_button)
        vbox.addWidget(self.predicted_button)
        vbox.addWidget(self.play_button)
        self.setLayout(vbox)

        # connect button signals to slots
        self.record_button.clicked.connect(self.record_audio)

        self.predicted_button.clicked.connect(self.predicted_audio)

        self.play_button.clicked.connect(self.play_audio)



        # set up recorded data buffer
        self.recorded_data = []

    # ////////////////

    def record_audio(self):
        # Record 1 second of audio
        self.record_button.setDisabled(True)
        self.play_button.setDisabled(True)
        self.predicted_button.setDisabled(True)
        duration = 1  # in seconds
        recording = sd.rec(int(duration * self.sample_rate), samplerate=self.sample_rate, channels=1)
        sd.wait()
        self.record_button.setDisabled(False)
        self.play_button.setDisabled(False)
        self.predicted_button.setDisabled(False)



        self.recording = recording


    def predicted_audio(self):

    

        # signal, sample_rate = librosa.load("./recording.wav")
        if len(self.recording) >= self.SAMPLES_TO_CONSIDER:
            # ensure consistency of the length of the signal
            signal = self.recording
            signal = signal.T
            signal = signal[:self.SAMPLES_TO_CONSIDER]

            # extract MFCCs
            MFCCs = librosa.feature.mfcc(signal, self.sample_rate, n_mfcc=13, n_fft=2048, hop_length=512)
            self.MFCCs = MFCCs.T
        else:
            print("Audio File length is not sataisfied")
            exit(0)
        
        self.MFCCs = self.MFCCs[np.newaxis, ..., np.newaxis]

        predictions = self.model.predict(self.MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mapping[predicted_index]
        # print(predicted_keyword)

        for tb in self.textboxes:
            if predicted_keyword.lower() == tb.text().lower():
                tb.setStyleSheet("QLineEdit { background-color: green; }")
            else:
                tb.setStyleSheet("QLineEdit { background-color: white; }")


    def play_audio(self):
        # Play back the recorded audio
        if hasattr(self, "recording"):
            sd.play(self.recording, self.sample_rate)
            print(self.recording)
            sd.wait()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    ex.show()
    sys.exit(app.exec_())
