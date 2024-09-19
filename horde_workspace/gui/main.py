import sys
import time
from concurrent.futures.thread import ThreadPoolExecutor

from PIL import Image
from PySide6.QtCore import Signal
from PySide6.QtGui import QPixmap, Qt
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QComboBox,
    QLineEdit,
    QWidget,
    QTabWidget,
    QLabel,
    QPushButton,
    QGridLayout,
    QHBoxLayout,
    QSlider,
)

from horde_workspace.classes.job import Job
from horde_workspace.data import LORAS, EMBEDDINGS, MODELS
from horde_workspace.gui.clipboard import (
    copy_image_to_clipboard,
    open_file_in_default_app,
)
from horde_workspace.processors import generate_images
from horde_workspace.workspace import Workspace


def new_slider() -> QSlider:
    slider = QSlider()
    slider.setOrientation(Qt.Orientation.Horizontal)
    slider.setMinimum(256 // 64)
    slider.setMaximum(2048 // 64)
    slider.setTickInterval(1)
    slider.setTickPosition(QSlider.TickPosition.TicksBelow)
    return slider


class WorkspaceWidget(QWidget):
    signal = Signal(int)

    def __init__(self, manager: "HordeWorkSpaceManager", workspace_name: str):
        super().__init__()

        self.tab_widget = manager.tabs

        self.workspace = Workspace("../../output/" + workspace_name)
        self.queue = 0
        self.kudos = 0
        self.cols = 6

        self.executor = ThreadPoolExecutor(10)

        self.signal.connect(self.on_image_generated)

        # Layout for the workspace
        layout = QVBoxLayout(self)

        # Prompt
        self.prompt = QLineEdit(self)
        self.prompt.setPlaceholderText("Enter prompt here")
        layout.addWidget(self.prompt)

        # Model
        options_layout = QHBoxLayout()
        self.model_dropdown = QComboBox(self)
        self.model_dropdown.addItems(list(MODELS.keys()))
        options_layout.addWidget(self.model_dropdown)

        # Lora
        self.lora_dropdown = QComboBox(self)
        self.lora_dropdown.addItems(["None"] + list(LORAS.keys()))
        options_layout.addWidget(self.lora_dropdown)

        # Embedding
        self.embedding_dropdown = QComboBox(self)
        self.embedding_dropdown.addItems(["None"] + list(EMBEDDINGS.keys()))
        options_layout.addWidget(self.embedding_dropdown)

        # Width and height
        width_label = QLabel("Width: ")
        self.width_slider = new_slider()
        self.width_slider.valueChanged.connect(
            lambda _: width_label.setText(f"Width: {self.width_slider.value() * 64}")
        )
        options_layout.addWidget(width_label)
        options_layout.addWidget(self.width_slider)

        height_label = QLabel("Height: ")
        self.height_slider = new_slider()
        self.height_slider.valueChanged.connect(
            lambda _: height_label.setText(f"Width: {self.height_slider.value() * 64}")
        )
        options_layout.addWidget(height_label)
        options_layout.addWidget(self.height_slider)

        # Close workspace
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close_tab)
        options_layout.addWidget(self.close_button)

        # Image input
        source_box = QHBoxLayout()
        self.source_image = QLineEdit(self)
        self.source_image.setPlaceholderText("Source image path")
        source_box.addWidget(self.source_image)

        self.denoising_strength_label = QLabel("Denoising strength: ")
        source_box.addWidget(self.denoising_strength_label)

        # Denoising strength
        self.denoising_strength = QSlider()
        self.denoising_strength.setOrientation(Qt.Orientation.Horizontal)
        self.denoising_strength.setMinimum(0)
        self.denoising_strength.setMaximum(100)
        self.denoising_strength.setTickInterval(10)
        self.denoising_strength.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.denoising_strength.setMaximumWidth(256)
        self.denoising_strength.valueChanged.connect(
            lambda _: self.denoising_strength_label.setText(
                f"Denoising strength: {self.denoising_strength.value()}"
            )
        )
        source_box.addWidget(self.denoising_strength)

        # Queue and kudos
        self.queue_label = QLabel("Queue: 0")
        source_box.addWidget(self.queue_label)
        self.kudos_label = QLabel("Kudos: 0")
        source_box.addWidget(self.kudos_label)

        layout.addLayout(source_box)

        # Widget inside the scroll area (for images and buttons)
        self.gallery = QWidget(self)
        self.gallery_layout = QGridLayout(self.gallery)

        layout.addLayout(options_layout)
        layout.addWidget(self.gallery)

        # Current job
        self.job = Job(
            prompt="",
            negprompt="",
            steps=30,
            width=1024,
            height=1024,
            model="AlbedoBase XL (SDXL)",
        )

        self.refresh_job()

        self.images = []
        self.refresh_images()

    def refresh_images(self):
        while self.gallery_layout.count():
            item = self.gallery_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        for i, image_data in enumerate(self.images):
            widget = QWidget()
            image_box = QVBoxLayout(widget)

            width = self.width() // self.cols - 32

            # Preview
            image_label = QLabel(f"Image {i + 1}", self)
            pixmap = QPixmap(image_data["file"])
            image_label.setPixmap(pixmap)
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_label.setStyleSheet("background-color: lightgray;")
            image_label.setScaledContents(True)
            image_label.setFixedSize(
                width, int(pixmap.height() * width / max(32, pixmap.width()))
            )
            image_label.mousePressEvent = lambda _, img=image_data: self.open_image(img)  # pyright: ignore [reportAttributeAccessIssue]

            # Button row under the image
            button_layout = QHBoxLayout()

            # Copy image
            btn = QPushButton("C", self)
            btn.clicked.connect(lambda _, img=image_data: self.copy_image(img))
            btn.setMaximumWidth(width // 5)
            btn.setToolTip("Copy image to clipboard")
            button_layout.addWidget(btn)

            btn = QPushButton("M", self)
            btn.clicked.connect(lambda _, img=image_data: self.remix_image(img))
            btn.setMaximumWidth(width // 5)
            btn.setToolTip("Remix image")
            button_layout.addWidget(btn)

            # Variate image
            btn = QPushButton("V", self)
            btn.clicked.connect(lambda _, img=image_data: self.variate_image(img))
            btn.setMaximumWidth(width // 5)
            btn.setToolTip("Variate image")
            button_layout.addWidget(btn)

            # Remove image
            btn = QPushButton("X", self)
            btn.clicked.connect(lambda _, img=image_data: self.remove_image(img))
            btn.setMaximumWidth(width // 5)
            btn.setToolTip("Remove image")
            button_layout.addWidget(btn)

            image_box.addWidget(image_label)
            image_box.addLayout(button_layout)
            image_box.addStretch()

            self.gallery_layout.addWidget(widget, i // self.cols, i % self.cols)

        self.gallery_layout.setColumnStretch(self.cols - 1, 1)
        self.gallery_layout.setRowStretch(len(self.images) // self.cols, 1)

    def close_tab(self):
        self.tab_widget.removeTab(self.tab_widget.currentIndex())

    # noinspection PyMethodMayBeStatic
    def open_image(self, image_data: dict):
        open_file_in_default_app(image_data["file"])

    # noinspection PyMethodMayBeStatic
    def copy_image(self, image_data: dict):
        copy_image_to_clipboard(image_data["file"])

    def remix_image(self, image_data: dict):
        self.job = image_data["job"].model_copy(deep=True)
        self.refresh_job()
        self.source_image.setText("")

    def variate_image(self, image_data: dict):
        self.job = image_data["job"].model_copy(deep=True)
        self.job.source_image = Image.open(image_data["file"])
        if self.job.denoising_strength == 1.0:
            self.job.denoising_strength = 0.75
        self.refresh_job()
        self.source_image.setText(image_data["file"])

    def remove_image(self, image_data: dict):
        for i, image in enumerate(self.images):
            if image["file"] == image_data["file"]:
                self.images.pop(i)
                break
        self.refresh_images()

    def refresh_job(self):
        self.prompt.setText(
            (self.job.prompt + " ### " + self.job.negprompt).strip().strip("#").strip()
        )
        self.model_dropdown.setCurrentText(str(self.job.model))
        self.lora_dropdown.setCurrentText(
            str(self.job.loras[0]) if self.job.loras else ""
        )
        self.embedding_dropdown.setCurrentText(
            str(self.job.tis[0]) if self.job.tis else ""
        )
        self.width_slider.setValue(self.job.width // 64)
        self.height_slider.setValue(self.job.height // 64)
        self.denoising_strength.setValue(int(self.job.denoising_strength * 100))

    def on_prompt_enter(self):
        parts = self.prompt.text().split("###", 1)
        self.job.prompt = parts[0].strip()
        self.job.negprompt = parts[1].strip() if len(parts) > 1 else ""
        self.job.model = MODELS[self.model_dropdown.currentText()]
        self.job.loras = (
            [LORAS[self.lora_dropdown.currentText()]]
            if self.lora_dropdown.currentText() != "None"
            else []
        )
        self.job.tis = (
            [EMBEDDINGS[self.embedding_dropdown.currentText()]]
            if self.embedding_dropdown.currentText() != "None"
            else []
        )
        self.job.width = self.width_slider.value() * 64
        self.job.height = self.height_slider.value() * 64

        if self.source_image.text():
            self.job.source_image = Image.open(self.source_image.text())
            self.job.denoising_strength = self.denoising_strength.value() / 100
        else:
            self.job.denoising_strength = 1.0

        # Start future
        self.queue += 1
        self.update_queue()
        self.executor.submit(
            self.generate_image, job=self.job.model_copy(deep=True), attempts=3
        ).add_done_callback(lambda result: self.signal.emit(result))

    def update_queue(self):
        self.queue_label.setText(f"Queue: {self.queue}")

    def update_kudos(self):
        self.kudos_label.setText(f"Kudos: {self.kudos}")

    def on_image_generated(self):
        self.queue -= 1
        self.update_queue()
        self.update_kudos()
        self.refresh_images()

    def generate_image(self, job: Job, attempts: int = 3):
        for _ in range(attempts):
            try:
                generation = generate_images(self.workspace, job)
                path = self.workspace.directory / self.workspace.save(
                    generation.get_image()
                )
                self.kudos += generation.kudos
                self.images.append(
                    {
                        "file": str(path.resolve()),
                        "job": job,
                    }
                )
                return
            except Exception as e:
                print(f"Error generating image: {e}")
                time.sleep(1)
                continue

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Enter, Qt.Key.Key_Return):
            self.on_prompt_enter()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_images()


class HordeWorkSpaceManager(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Horde Workspace")

        # Central widget (tabs)
        self.tabs = QTabWidget(self)
        self.setCentralWidget(self.tabs)

        # Layout for the control tab
        layout = QHBoxLayout()

        # Workplace name
        self.workspace_name = QLineEdit(self)
        self.workspace_name.setPlaceholderText("default")
        layout.addWidget(self.workspace_name)

        # Add the button to the layout
        self.add_tab_button = QPushButton("New Tab")
        self.add_tab_button.setFixedWidth(100)
        self.add_tab_button.clicked.connect(self.add_workspace)
        layout.addWidget(self.add_tab_button)

        # Create a main widget to hold the layout
        control_widget = QWidget()
        control_widget.setLayout(layout)
        self.tabs.addTab(control_widget, "New")

        # Default workspace
        self.add_workspace()

    def add_workspace(self):
        workspace_name = self.workspace_name.text() or "default"
        new_workspace = WorkspaceWidget(self, workspace_name)
        index = self.tabs.addTab(new_workspace, workspace_name)
        self.tabs.setCurrentIndex(index)

    def close_current_workspace(self):
        current_index = self.tabs.currentIndex()
        if current_index != -1:
            self.tabs.removeTab(current_index)

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if not self.isMaximized():
            self.showMaximized()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            tab = self.tabs.currentWidget()
            if isinstance(tab, WorkspaceWidget):
                tab.source_image.setText(file_path)
                if tab.job.denoising_strength == 1.0:
                    tab.job.denoising_strength = 0.75
                tab.refresh_job()


def main():
    app = QApplication(sys.argv)

    try:
        # noinspection PyUnresolvedReferences
        import qdarktheme

        qdarktheme.setup_theme()
    except ImportError:
        pass

    window = HordeWorkSpaceManager()
    window.showMaximized()
    window.show()

    window.installEventFilter(window)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
