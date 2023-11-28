# LCM_i2i_PoC
LCM-LoRAを用いた高速i2iの検証用リポジトリ

# 前提環境
以下をインストールしておくこと<br>
git: [git](https://git-scm.com/downloads)<br>
Python: [3.8.10](https://www.python.org/downloads/release/python-3810/) (3.10でも起動確認済み)  
CUDA Toolkit: [11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)<br>

# 使い方
①適当なディレクトリでリポジトリをgit clone<br>
```
cd C:/
git clone --branch test https://github.com/tori29umai0123/LCM_i2i_PoC.git
```
②install.ps1を右クリック→PowerShellで実行（5分位かかります）<br>
③realtime_i2i.ps1を右クリック→PowerShellで実行<br>
④各種設定してからキャプチャ開始ボタンをクリック。範囲設定後、画像が自動で生成される。<br>
⑤『Window Capture』（生成画像が表示されるウインドウ）がアクティブ時、『P』キーを押すとクリップボードに画像が貼り付けられるので適宜使用してください。

# 設定例
- **LCMモデル名**: `latent-consistency/lcm-lora-sdv1-5`
- **生成モデル名**: `852wa/SDHK`
- **VAEモデルパス**: `C:\stable-diffusion-webui\models\VAE\kl-f8-anime2.ckpt`
- **LoRAモデルパス**: `C:\stable-diffusion-webui\models\Lora\style-lineart.safetensors`
- **LoRAstrength**: `1.0`
- **プロンプト**: `1girl, line art, chibi, full body`
- **strength**: `0.5`
- **num_inference_steps**: `8`
