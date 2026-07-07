import sys

from PyQt6.QtWidgets import QApplication

from .application.controller import AppController
from .ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    controller = AppController()
    app.aboutToQuit.connect(controller.shutdown)

    window = MainWindow(controller=controller)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
