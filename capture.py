from pynput import mouse
from PIL import ImageGrab

import tkinter as tk

class ScreenCapture:

    def __init__(self):
        self.root = tk.Tk()
        self.root.attributes('-fullscreen', True)
        self.root.attributes('-topmost', True)
        self.root.attributes('-alpha', 0.5)
        self.root.configure(bg='white')
        self.root.bind('<Escape>', lambda e: self.root.quit())

        self.start_position = None
        self.selection = None

    def on_click(self, x, y, button, pressed):
        if pressed:
            self.start_position = (x, y)
            self.selection = tk.Canvas(self.root, cursor="cross", bg='black')
            self.selection.place(x=x, y=y, width=1, height=1)
        else:
            self.root.quit()

    def on_drag(self, x, y):
        if self.start_position:
            self.selection.place_configure(width=max(1, x - self.start_position[0]), height=max(1, y - self.start_position[1]))


    def capture(self):
        screenshot = ImageGrab.grab(bbox=(self.x0, self.y0, self.x1, self.y1))
        #screenshot = ImageGrab.grab(bbox=(self.start_position[0], self.start_position[1], self.end_position[0], self.end_position[1]))
        return screenshot


    def listen(self):
        # マウスリスナーの開始
        listener = mouse.Listener(on_click=self.on_click, on_move=self.on_drag)
        listener.start()

        self.root.mainloop()
        listener.stop()
        self.x0, self.y0 = self.start_position
        self.x1, self.y1 = self.selection.winfo_x() + self.selection.winfo_width(), self.selection.winfo_y() + self.selection.winfo_height()
