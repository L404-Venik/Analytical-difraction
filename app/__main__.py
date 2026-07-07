import sys
from pathlib import Path

from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QApplication

from .application.controller import AppController
from .ui.main_window import MainWindow

APP_USER_MODEL_ID = "sphere-diffraction.gui"


def load_icon() -> QIcon:
    icon = QIcon()
    resources = Path(__file__).parent / "resources"
    for png in sorted(resources.glob("icon_*.png")):
        icon.addFile(str(png))
    return icon


def main():
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
            APP_USER_MODEL_ID
        )

    app = QApplication(sys.argv)
    icon = load_icon()
    if not icon.isNull():
        app.setWindowIcon(icon)

    controller = AppController()
    app.aboutToQuit.connect(controller.shutdown)

    window = MainWindow(controller=controller)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
