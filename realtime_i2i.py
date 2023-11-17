import cv2
import numpy as np
import pygetwindow as gw
import pyautogui

from diffusers import AutoPipelineForImage2Image, UNet2DConditionModel, LCMScheduler
from diffusers.utils import load_image
import torch

lcm_lora_id = "latent-consistency/lcm-lora-sdv1-5"

#pipe = DiffusionPipeline.from_pretrained(model_id, variant="fp16", torch_dtype=torch.float16).to("cuda")
pipe = AutoPipelineForImage2Image.from_pretrained(
    "852wa/SDHK", torch_dtype=torch.float16, use_safetensors=True
).to("cuda")


pipe.load_lora_weights(lcm_lora_id)
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

if pipe.safety_checker is not None:
    pipe.safety_checker = lambda images, **kwargs: (images, [False])


prompt = "1girl, black hair, school uniform, 8k"


# 画像を表示するウィンドウの作成
cv2.namedWindow("Active Window Capture", cv2.WINDOW_NORMAL)

while True:
    # アクティブなウィンドウを取得
    active_window = gw.getActiveWindow()
    if active_window:
        # アクティブなウィンドウの位置とサイズを取得
        x, y, width, height = active_window.left, active_window.top, active_window.width, active_window.height

        # 指定された領域のスクリーンショットを取得
        screenshot = pyautogui.screenshot(region=(x, y, width, height))
        img_np = np.array(screenshot)

        

        generator = torch.Generator("cuda").manual_seed(2500)
        img = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        

        img = pipe(
            prompt=prompt,
            image=img,
            num_inference_steps=8,
            guidance_scale=1,
            generator=generator
        ).images[0]

        # PILイメージをnumpy配列に変換
        img = np.array(img)

        # OpenCVでは色の順番がBGRなので、RGBからBGRに変換
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # PILの画像をOpenCVの形式に変換
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


        # キャプチャした画像を同じウィンドウで更新して表示
        cv2.imshow("Active Window Capture", img)

        # 'q'を押したらループを抜ける
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()

