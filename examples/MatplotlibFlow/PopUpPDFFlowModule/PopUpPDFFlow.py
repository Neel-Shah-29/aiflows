import os
import tkinter as tk
from PIL import Image, ImageTk

from aiflows.base_flows import AtomicFlow
from aiflows.utils import logging

log = logging.get_logger(__name__)

class PopupPngFlow(AtomicFlow):
    """
    PopupPngFlow is a flow that displays a PNG image in a popup window.
    It's designed to work with images, specifically matplotlib plots saved as PNG.

    *Configuration Parameters*:
    - `name` (str): The name of the flow.
    - `description` (str): A description of the flow.

    *Input Interface*:
    - `png_file_path` (str): The file path of the PNG image to be displayed.

    *Output Interface*:
    - No output, as this is a display-only flow.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, input_data):
        file_path = input_data.get('png_file_path')
        if not file_path or not os.path.exists(file_path):
            log.error("PNG file path is invalid or file does not exist")
            return

        self._display_image(file_path)

    def _display_image(self, file_path):
        window = tk.Tk()
        window.title("Image Display")

        img = Image.open(file_path)
        img = ImageTk.PhotoImage(img)
        panel = tk.Label(window, image=img)
        panel.pack(side="bottom", fill="both", expand="yes")

        window.mainloop()
