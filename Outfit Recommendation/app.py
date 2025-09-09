import sys
import cv2
import os
import pandas as pd
import random
import requests
from io import BytesIO
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QRadioButton, QButtonGroup, QFileDialog, QComboBox,
    QStackedWidget
)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import QTimer, Qt
from PIL import Image
from math import sqrt

#AI LOGIC
skin_tone_to_color_mapping = {
    "#373028": ["navy blue", "black", "charcoal", "burgundy", "maroon", "olive", "rust", "gold", "cream", "peach"],
    "#422811": ["navy blue", "brown", "khaki", "olive", "maroon", "mustard", "teal", "tan", "rust", "burgundy"],
    "#513B2E": ["cream", "beige", "olive", "burgundy", "red", "orange", "mustard", "bronze", "teal", "peach"],
    "#6F503C": ["beige", "brown", "green", "khaki", "cream", "peach", "lime green", "olive", "maroon", "rust", "mustard"],
    "#81654F": ["beige", "off white", "sea green", "cream", "lavender", "mauve", "burgundy", "yellow", "lime green"],
    "#9D7A54": ["olive", "khaki", "yellow", "sea green", "turquoise blue", "coral", "white", "gold", "peach"],
    "#BEA07E": ["coral", "sea green", "turquoise blue", "pink", "lavender", "rose", "white", "peach", "teal", "fluorescent green"],
    "#E5C8A6": ["turquoise blue", "peach", "teal", "pink", "red", "rose", "off white", "white", "cream", "gold", "yellow"],
    "#E7C1B8": ["pink", "rose", "peach", "white", "off white", "beige", "lavender", "teal", "fluorescent green"],
    "#F3DAD6": ["white", "cream", "peach", "pink", "rose", "lavender", "mustard", "lime green", "light blue", "fluorescent green"],
    "#FBF2F3": ["peach", "lavender", "pink", "white", "off white", "rose", "light blue", "sea green", "fluorescent green", "silver", "cream", "tan"]
}

images_df = pd.read_csv("images.csv")
fashion_df = pd.read_csv("styles.csv", on_bad_lines='skip')

#koversi warna HEX ke RGB
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
#cari warna kulit terdekat dari input RGB
def get_closest_skin_tone(rgb_input, mapping):
    min_dist = float("inf")
    closest_hex = None
    for hex_code in mapping.keys():
        rgb_ref = hex_to_rgb(hex_code)
        dist = sqrt(sum((a - b) ** 2 for a, b in zip(rgb_input, rgb_ref)))
        if dist < min_dist:
            min_dist = dist
            closest_hex = hex_code
    return closest_hex


#deteksi warna kulit dari foto selfie
def detect_skin_tone_from_selfie(image_path, skin_tone_to_color_mapping):
    image = cv2.imread(image_path)
    if image is None:
        return None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None, None

    (x, y, w, h) = faces[0]
    face = image[y:y+h, x:x+w]
    patch = face[h//4:3*h//4, w//4:3*w//4]
    avg_color = patch.reshape(-1, 3).mean(axis=0)
    b, g, r = avg_color
    rgb = (int(r), int(g), int(b))
    closest_hex = get_closest_skin_tone(rgb, skin_tone_to_color_mapping)
    recommended_colors = skin_tone_to_color_mapping.get(closest_hex)
    return closest_hex, [c.lower() for c in recommended_colors]

#membuat outfit berdasarkan warna kulit dan memfilter outfit yang dihasilkan
def generate_outfits(df, gender, color_pool, num_outfits=3):
    outfits = []
    color_pool = [c.lower() for c in color_pool]

    filtered = df[
        (df['gender'].str.lower().isin([gender.lower(), "unisex"])) &
        (df['baseColour'].apply(lambda c: any(color in str(c).lower() for color in color_pool))) &
        (df['usage'].str.lower() != "na") &
        (~df['articleType'].str.lower().str.contains("swimwear", na=False))
    ]

    for _ in range(num_outfits):
        topwears = filtered[filtered['subCategory'].str.lower() == "topwear"]
        bottomwears = filtered[filtered['subCategory'].str.lower() == "bottomwear"]
        shoes = filtered[filtered['subCategory'].str.lower().str.contains("shoe", na=False)]
        accessories = filtered[filtered['masterCategory'].str.lower() == "accessories"]

        outfit = {
            "topwear": topwears.sample(1).iloc[0].to_dict() if not topwears.empty else {"productDisplayName": "Not available", "id": -1},
            "bottomwear": bottomwears.sample(1).iloc[0].to_dict() if not bottomwears.empty else {"productDisplayName": "Not available", "id": -1},
            "shoes": shoes.sample(1).iloc[0].to_dict() if not shoes.empty else {"productDisplayName": "Not available", "id": -1},
            "accessory": accessories.sample(1).iloc[0].to_dict() if not accessories.empty else None
        }

        outfits.append(outfit)

    return outfits
#menggabungkan deteksi kulit dan pembuatan outfit
def run_selfie_analysis_and_generate_outfits(selfie_path, gender):
    hex_code, color_pool = detect_skin_tone_from_selfie(selfie_path, skin_tone_to_color_mapping)
    if hex_code and color_pool:
        outfits = generate_outfits(fashion_df, gender, color_pool, num_outfits=3)
        return hex_code, color_pool, outfits
    else:
        return None, None, []

#pengambilan gambar
images_df = pd.read_csv("images.csv")

def get_image_link(item_id):
    item_filename = f"{str(item_id)}.jpg"

    if 'filename' in images_df.columns:
        match = images_df[images_df['filename'] == item_filename]
    elif 'id' in images_df.columns:
        match = images_df[images_df['id'] == item_id]
    else:
        print("❌ No valid ID column found in images.csv.")
        return None

    if not match.empty:
        return match.iloc[0]['link']
    return None


#GUI
#halaman utama
class HomePage(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.setWindowTitle("Outfit Generator")
        self.setFixedSize(1280, 720)
        self.setStyleSheet("background-color: #2e2e2e; color: white;")

        self.available_cameras = self.get_available_cameras()
        self.camera_index = 0
        self.capture = cv2.VideoCapture(self.camera_index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 450)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.init_ui()
        self.timer.start(30)

    def get_available_cameras(self):
        cameras = []
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.read()[0]:
                cameras.append(i)
                cap.release()
        return cameras

    def init_ui(self):
        self.layout = QVBoxLayout()

        self.title = QLabel("OUTFIT GENERATOR")
        self.title.setFont(QFont("Arial", 20, QFont.Bold))
        self.title.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.title)

        self.camera_selector = QComboBox()
        self.camera_selector.setStyleSheet("background-color: lightgray; color: black;")
        for i in self.available_cameras:
            self.camera_selector.addItem(f"Camera {i}")
        self.camera_selector.currentIndexChanged.connect(self.change_camera)
        self.layout.addWidget(self.camera_selector, alignment=Qt.AlignCenter)

        self.camera_label = QLabel()
        self.layout.addWidget(self.camera_label, alignment=Qt.AlignCenter)

        bottom_layout = QVBoxLayout()
        bottom_layout.setAlignment(Qt.AlignCenter)

        gender_layout = QHBoxLayout()
        self.gender_group = QButtonGroup(self)
        self.male_radio = QRadioButton("Male")
        self.female_radio = QRadioButton("Female")
        self.gender_group.addButton(self.male_radio)
        self.gender_group.addButton(self.female_radio)
        self.male_radio.toggled.connect(self.check_gender_selection)
        self.female_radio.toggled.connect(self.check_gender_selection)
        gender_layout.addWidget(self.male_radio)
        gender_layout.addWidget(self.female_radio)

        bottom_layout.addLayout(gender_layout)

        self.capture_btn = QPushButton("SELECT GENDER FIRST")
        self.capture_btn.setStyleSheet("background-color: #555555; color: #aaaaaa;")
        self.capture_btn.clicked.connect(self.capture_image)
        self.capture_btn.setEnabled(False)
        bottom_layout.addWidget(self.capture_btn)

        self.layout.addLayout(bottom_layout)
        self.setLayout(self.layout)

    def check_gender_selection(self):
        if self.male_radio.isChecked() or self.female_radio.isChecked():
            self.capture_btn.setEnabled(True)
            self.capture_btn.setText("TAKE PICTURE")
            self.capture_btn.setStyleSheet("background-color: lightgray; color: black;")
        else:
            self.capture_btn.setEnabled(False)
            self.capture_btn.setText("SELECT GENDER FIRST")
            self.capture_btn.setStyleSheet("background-color: #555555; color: #aaaaaa;")

    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_qt = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(convert_to_qt).scaled(800, 450)
            self.camera_label.setPixmap(pixmap)

    def change_camera(self, index):
        self.camera_index = self.available_cameras[index]
        self.capture.release()
        self.capture = cv2.VideoCapture(self.camera_index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 450)
    
    def capture_image(self):
        ret, frame = self.capture.read()
        if not ret:
            print("❌ Failed to capture image.")
            return

        save_path = os.path.join(os.getcwd(), "selfie.png")
        cv2.imwrite(save_path, frame)
        gender = "Male" if self.male_radio.isChecked() else "Female"
        print("✅ Selfie captured and saved as selfie.png")

        # 🔍 Analyze selfie and generate outfits
        hex_code, color_pool, outfits = run_selfie_analysis_and_generate_outfits(save_path, gender)

        if not outfits:
            print("❌ Failed to generate outfits.")
            return

        # 👉 Go to loading screen
        loading_page = LoadingPage(save_path, gender, self.stacked_widget, outfits)
        self.stacked_widget.addWidget(loading_page)
        self.stacked_widget.setCurrentWidget(loading_page)

        # ✅ Update loading results
        QTimer.singleShot(1000, lambda: loading_page.update_results(hex_code, color_pool))

    def closeEvent(self, event):
        self.capture.release()
        self.timer.stop()
        event.accept()

#halaman warna rekomendasi untuk warna kulit
class LoadingPage(QWidget):
    def __init__(self, selfie_path, gender, stacked_widget, outfits):
        super().__init__()
        self.setFixedSize(1280, 720)
        self.setStyleSheet("background-color: #2e2e2e; color: white;")

        self.selfie_path = selfie_path
        self.gender = gender
        self.stacked_widget = stacked_widget
        self.outfits = outfits

        layout = QVBoxLayout()

        selfie_and_text_layout = QHBoxLayout()
        selfie_and_text_layout.setAlignment(Qt.AlignCenter)

        selfie_label = QLabel()
        pixmap = QPixmap(selfie_path).scaled(240, 135, Qt.KeepAspectRatio)
        selfie_label.setPixmap(pixmap)
        selfie_and_text_layout.addWidget(selfie_label)

        title_and_info_layout = QVBoxLayout()
        title_and_info_layout.setAlignment(Qt.AlignVCenter)

        title = QLabel("tone and color recommendation")
        title.setFont(QFont("Arial", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title_and_info_layout.addWidget(title)

        self.hex_label = QLabel("🎨 Skin Tone HEX: #------")
        self.color_label = QLabel("🎨 Recommended Colors: Loading...")
        self.hex_label.setFont(QFont("Arial", 14))
        self.color_label.setFont(QFont("Arial", 14))
        title_and_info_layout.addWidget(self.hex_label)
        title_and_info_layout.addWidget(self.color_label)

        selfie_and_text_layout.addLayout(title_and_info_layout)
        layout.addLayout(selfie_and_text_layout)

        self.continue_button = QPushButton("CONTINUE TO OUTFITS")
        self.continue_button.setEnabled(False)
        self.continue_button.setStyleSheet("background-color: #555555; color: #aaaaaa;")
        self.continue_button.clicked.connect(self.go_to_outfit_page)
        layout.addWidget(self.continue_button, alignment=Qt.AlignCenter)

        self.setLayout(layout)

    def update_results(self, hex_code, recommended_colors):
        self.hex_label.setText(f"🎨 Skin Tone HEX: {hex_code}")
        self.color_label.setText("🎨 Recommended Colors: " + ", ".join(recommended_colors))
        self.continue_button.setEnabled(True)
        self.continue_button.setStyleSheet("background-color: lightgray; color: black;")

    def go_to_outfit_page(self):
        outfit_pages = [
            OutfitPage(self.selfie_path, self.outfits[0], 1, self.stacked_widget),
            OutfitPage(self.selfie_path, self.outfits[1], 2, self.stacked_widget),
            OutfitPage(self.selfie_path, self.outfits[2], 3, self.stacked_widget, is_last=True)
        ]
        for page in outfit_pages:
            self.stacked_widget.addWidget(page)
        self.stacked_widget.setCurrentWidget(outfit_pages[0])

#halaman hasil outfit
class OutfitPage(QWidget):
    def __init__(self, selfie_path, outfit, index, stacked_widget, is_last=False):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.setStyleSheet("background-color: #2e2e2e; color: white;")
        self.setFixedSize(1280, 720)

        layout = QVBoxLayout()

        header = QLabel(f"OUTFIT {index}")
        header.setFont(QFont("Arial", 20, QFont.Bold))
        header.setAlignment(Qt.AlignCenter)

        selfie_label = QLabel()
        if os.path.exists(selfie_path):
            pixmap = QPixmap(selfie_path).scaled(200, 200, Qt.KeepAspectRatio)
            selfie_label.setPixmap(pixmap)

        item_layout = QHBoxLayout()
        for key in ["topwear", "bottomwear", "shoes", "accessory"]:
            item = outfit.get(key)
            if item:
                item_vbox = QVBoxLayout()
                label = QLabel(item['productDisplayName'])
                label.setAlignment(Qt.AlignCenter)
                img_label = QLabel()
                link = get_image_link(item['id'])
                if link:
                    try:
                        response = requests.get(link)
                        if response.status_code == 200:
                            image = QImage.fromData(response.content)
                            img_label.setPixmap(QPixmap.fromImage(image).scaled(150, 150, Qt.KeepAspectRatio))
                    except:
                        pass
                img_label.setAlignment(Qt.AlignCenter)
                item_vbox.addWidget(img_label)
                item_vbox.addWidget(label)
                item_layout.addLayout(item_vbox)

        btn_layout = QHBoxLayout()
        if index > 1:
            prev_btn = QPushButton("Previous")
            prev_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(self.stacked_widget.currentIndex() - 1))
            btn_layout.addWidget(prev_btn)

        if is_last:
            next_btn = QPushButton("Exit")
            next_btn.clicked.connect(lambda: QApplication.quit())
        else:
            next_btn = QPushButton("Next")
            next_btn.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(self.stacked_widget.currentIndex() + 1))
        btn_layout.addWidget(next_btn)

        top_layout = QHBoxLayout()
        top_layout.addWidget(selfie_label)
        top_layout.addStretch()
        top_layout.addWidget(header)
        top_layout.addStretch()

        layout.addLayout(top_layout)
        layout.addSpacing(30)
        layout.addLayout(item_layout)
        layout.addSpacing(20)
        layout.addLayout(btn_layout)

        self.setLayout(layout)


    def update_outfit(self):
        outfit = self.outfits[self.current_index]
        self.title_label.setText(f"OUTFIT {self.current_index + 1}")
        self.topwear.setText(f"👕 Topwear: {outfit['topwear']['productDisplayName']}")
        self.bottomwear.setText(f"👖 Bottomwear: {outfit['bottomwear']['productDisplayName']}")
        self.shoes.setText(f"👟 Shoes: {outfit['shoes']['productDisplayName']}")
        if outfit['accessory']:
            self.accessory.setText(f"🎒 Accessory: {outfit['accessory']['productDisplayName']}")
        else:
            self.accessory.setText("🎒 Accessory: Not available")

        self.prev_button.setEnabled(self.current_index > 0)
        if self.current_index == len(self.outfits) - 1:
            self.next_button.setText("Exit")
        else:
            self.next_button.setText("Next")

    def prev_outfit(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_outfit()

    def next_or_exit(self):
        if self.current_index < len(self.outfits) - 1:
            self.current_index += 1
            self.update_outfit()
        else:
            QApplication.quit()




if __name__ == '__main__':
    app = QApplication(sys.argv)
    stacked = QStackedWidget()
    home = HomePage(stacked)
    stacked.addWidget(home)
    stacked.setCurrentWidget(home)
    stacked.setFixedSize(1280, 720)
    stacked.setWindowTitle("Outfit Generator")
    stacked.setStyleSheet("background-color: #2e2e2e;")
    stacked.setWindowFlags(stacked.windowFlags() & ~Qt.WindowMaximizeButtonHint)
    stacked.show()
    sys.exit(app.exec_())
