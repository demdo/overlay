import sys
import pydicom
import matplotlib.pyplot as plt
from PySide6.QtWidgets import QApplication, QFileDialog


# ---------- Helper ----------
def get_qt_app():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    return app


def print_dicom_tags(ds):
    print("\n" + "=" * 80)
    print("DICOM TAGS")
    print("=" * 80)

    for elem in ds:
        print(elem)

    print("=" * 80 + "\n")


def show_dicom_image(ds):
    if 'PixelData' in ds:
        img = ds.pixel_array

        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.title("DICOM Image")
        plt.axis('off')
        plt.show()
    else:
        print("No pixel data found.")


# ---------- Main ----------
def main():
    app = get_qt_app()

    file_path, _ = QFileDialog.getOpenFileName(
        None,
        "Select DICOM file",
        "",
        "DICOM files (*.dcm *.IMA);;All files (*)"
    )

    if not file_path:
        print("No file selected.")
        return

    print(f"\nSelected file: {file_path}")

    # DICOM laden
    ds = pydicom.dcmread(file_path)

    # Tags ausgeben
    print_dicom_tags(ds)

    # Bild anzeigen
    show_dicom_image(ds)


if __name__ == "__main__":
    main()