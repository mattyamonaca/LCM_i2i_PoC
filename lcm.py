from pynput import keyboard
import pyperclip
import cv2
import numpy as np
import pygetwindow as gw

from diffusers import AutoPipelineForImage2Image, UNet2DConditionModel, LCMScheduler, AutoencoderKL
from diffusers.utils import load_image
import torch
from PIL import Image
import pygetwindow as gw
import io
import win32clipboard
from PIL import Image

def send_image_to_clipboard(image):
    output = io.BytesIO()
    image.convert("RGB").save(output, "BMP")
    data = output.getvalue()[14:]  # BMPファイルヘッダを削除
    output.close()

    win32clipboard.OpenClipboard()  # クリップボードを開く
    win32clipboard.EmptyClipboard()  # クリップボードを空にする
    win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)  # クリップボードに画像を設定
    win32clipboard.CloseClipboard()  # クリップボードを閉じる


def get_pipe(config):
    vae_model_path = config.vae_model_path.get()
    vae_model_path = vae_model_path.replace("\\", "/")
    LoRA_model_path = config.LoRA_model_path.get()
    LoRA_model_path = LoRA_model_path.replace("\\", "/")

    if config.vae_model_path.get() != "":
        pipe = AutoPipelineForImage2Image.from_pretrained(
        config.generation_model_name.get(), torch_dtype=torch.float16, use_safetensors=True,
        vae = AutoencoderKL.from_single_file(vae_model_path, torch_dtype=torch.float16)
        ).to("cuda")
    else:
        pipe = AutoPipelineForImage2Image.from_pretrained(
        config.generation_model_name.get(), torch_dtype=torch.float16, use_safetensors=True,
        ).to("cuda")

    if config.LoRA_model_path.get() != "":
        pipe.load_lora_weights(LoRA_model_path, adapter_name="LoRA")
        pipe.load_lora_weights(config.lcm_model_name.get(), adapter_name="lcm")
        pipe.set_adapters(["LoRA", "lcm"], adapter_weights=[config.LoRAstrength_value, 1.0])
        pipe.fuse_lora()
    else:
        pipe.load_lora_weights(config.lcm_model_name.get(), adapter_name="lcm")

    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    if pipe.safety_checker is not None:
        pipe.safety_checker = lambda images, **kwargs: (images, [False])

    return pipe


def on_press(key):
    global img  # imgをグローバル変数として使用します
    if hasattr(key, 'char') and key.char == 'p':  # 'p'キーが押されたかチェック
        # OpenCVのBGR形式からPILのRGB形式に変換し、クリップボードにコピーする
        if isinstance(img, np.ndarray):
            try:
                # RGB形式に変換する前に、imgがNumPy配列であることを確認する
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                send_image_to_clipboard(pil_img)
            except Exception as e:
                print(f"An error occurred: {e}")
        else:
            print("The image is not in the correct format. img must be a NumPy array.")

def LCM_run(config, pipe):
    global img  # imgをグローバル変数として使用します

    window_name = "Window Capture"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    config.running = True
    listener = keyboard.Listener(on_press=on_press)
    listener.start()  # キーボードリスナーを開始

    try:
        while config.running:
            screenshot = config.screen_capture.capture()
            input_img_np = np.array(screenshot)

            generator = torch.Generator("cuda").manual_seed(2500)
            input_img = Image.fromarray(input_img_np)

            img_pil = pipe(
                strength=config.strength_value,
                prompt=config.prompt.get(),
                image=input_img,
                num_inference_steps=config.num_inference_steps_value,
                guidance_scale=1,
                generator=generator
            ).images[0]

            img = np.array(img_pil)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imshow(window_name, img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
        listener.stop()  # キーボードリスナーを停止する
        listener.join()  # リスナーの完全な停止を待つ# キーボードリスナーを停止
