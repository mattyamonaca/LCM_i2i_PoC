from diffusers.utils import load_image
import tkinter as tk
from tkinter import ttk
import threading

from lcm import get_pipe, LCM_run
from capture import ScreenCapture

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
