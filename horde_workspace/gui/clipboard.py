import io
import os
import subprocess

from PIL import Image


def copy_image_to_clipboard(path: str):
    if os.name == "nt":  # Windows
        # noinspection PyUnresolvedReferences
        import win32clipboard  # pyright: ignore [reportMissingModuleSource]

        # Convert image to BMP format for Windows clipboard
        output = io.BytesIO()
        Image.open(path).save(output, format="BMP")
        bmp_data = output.getvalue()[14:]  # Skip BMP header

        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32clipboard.CF_BITMAP, bmp_data)
        win32clipboard.CloseClipboard()
    elif os.name == "posix":
        if os.uname().sysname == "Darwin":  # macOS
            # TODO: Implement macOS clipboard
            pass
        else:
            os.system("xclip -selection clipboard -t image/webp -i " + path)


def open_file_in_default_app(file_path):
    if os.name == "nt":  # Windows
        os.startfile(file_path)
    elif os.name == "posix":  # macOS or Linux
        if (
            subprocess.run(["uname"], capture_output=True).stdout.decode().strip()
            == "Darwin"
        ):  # macOS
            subprocess.run(["open", file_path])
        else:  # Linux
            subprocess.run(["xdg-open", file_path])
