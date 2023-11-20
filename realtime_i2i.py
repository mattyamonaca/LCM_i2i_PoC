import cv2
import numpy as np
import pygetwindow as gw
import pyautogui

from diffusers import AutoPipelineForImage2Image, UNet2DConditionModel, LCMScheduler
from diffusers.utils import load_image
import torch
from PIL import Image

from pynput import mouse
from PIL import ImageGrab
import pygetwindow as gw

import tkinter as tk
from tkinter import ttk

import threading

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

def LCM_run(config, pipe):
    cv2.namedWindow("Window Capture", cv2.WINDOW_NORMAL)
    config.running = True
    while config.running:
        # アクティブなウィンドウを取得
        screenshot = config.screen_capture.capture()
        img_np = np.array(screenshot)

        generator = torch.Generator("cuda").manual_seed(2500)
        #img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_np)
        

        img = pipe(
            strength=config.strength_value,
            prompt=config.prompt.get(),
            image=img,
            num_inference_steps=config.num_inference_steps_value,
            guidance_scale=1,
            generator=generator
        ).images[0]

        # PILイメージをnumpy配列に変換
        img = np.array(img)

        # OpenCVでは色の順番がBGRなので、RGBからBGRに変換
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # キャプチャした画像を同じウィンドウで更新して表示
        cv2.imshow("Window Capture", img)

        # 'q'を押したらループを抜ける
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



    cv2.destroyAllWindows()

def get_pipe(config):
    pipe = AutoPipelineForImage2Image.from_pretrained(
        config.generation_model_name.get(), torch_dtype=torch.float16, use_safetensors=True
    ).to("cuda")


    pipe.load_lora_weights(config.lcm_model_name.get())
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    if pipe.safety_checker is not None:
        pipe.safety_checker = lambda images, **kwargs: (images, [False])

    return pipe



class ConfigWindow:
    def __init__(self):
        master = tk.Tk()
        self.run_thread = None
        self.running = False
        self.master = master
        master.title("Configuration")
        master.geometry("400x500")  # ウィンドウサイズを設定
        

        style = ttk.Style()
        style.configure("TLabel", font=("Arial", 12))
        style.configure("TEntry", padding=5)
        style.configure("TButton", padding=5, font=("Arial", 10))

        # LCMモデル名
        ttk.Label(master, text="LCMモデル名").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.lcm_model_name = tk.StringVar(value="latent-consistency/lcm-lora-sdv1-5")
        ttk.Entry(master, textvariable=self.lcm_model_name, width=30).grid(row=0, column=1, padx=10, pady=10)

        # 生成モデル名
        ttk.Label(master, text="生成モデル名").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.generation_model_name = tk.StringVar(value="852wa/SDHK")
        ttk.Entry(master, textvariable=self.generation_model_name, width=30).grid(row=1, column=1, padx=10, pady=10)

        # プロンプト
        ttk.Label(master, text="プロンプト").grid(row=2, column=0, padx=10, pady=10, sticky="w")
        self.prompt = tk.StringVar()
        ttk.Entry(master, textvariable=self.prompt, width=30).grid(row=2, column=1, padx=10, pady=10)

        # strength
        ttk.Label(master, text="strength").grid(row=3, column=0, padx=10, pady=10, sticky="w")
        self.strength = tk.StringVar(value=0.75)
        ttk.Entry(master, textvariable=self.strength, width=30).grid(row=3, column=1, padx=10, pady=10)
 
        # num_inference_steps
        ttk.Label(master, text="num_inference_steps").grid(row=4, column=0, padx=10, pady=10, sticky="w")
        self.num_inference_steps = tk.StringVar(value=8)
        ttk.Entry(master, textvariable=self.num_inference_steps, width=30).grid(row=4, column=1, padx=10, pady=10)
 

        #画面キャプチャ
        capture_button = ttk.Button(master, text="キャプチャ開始", command=self.capture_screen)
        capture_button.grid(row=5, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        #パラメータ更新
        capture_button = ttk.Button(master, text="パラメータ更新", command=self.update_param)
        capture_button.grid(row=6, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

    def update_param(self):
        self.num_inference_steps_value = int(self.num_inference_steps.get())
        self.strength_value = float(self.strength.get())


    def capture_screen(self):
        if self.run_thread is not None:
            self.running = False
            self.run_thread.join()

        self.screen_capture = ScreenCapture()
        self.screen_capture.listen()
        self.screen_capture.root.destroy()
        self.update_param()
        pipe = get_pipe(self)

        self.running = True
        self.run_thread = threading.Thread(target=LCM_run, args=(self, pipe))
        self.run_thread.start()


    def open(self):
        self.master.mainloop()



config = ConfigWindow()
config.open()
