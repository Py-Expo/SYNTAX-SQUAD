<div align="center">
<h2>Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data</h2>


## News

* **2024-02-27:** Depth Anything is accepted by CVPR 2024.
* **2024-02-05:** [Depth Anything Gallery](./gallery.md) is released. Thank all the users!
* **2024-02-02:** Depth Anything serves as the default depth processor for [InstantID](https://github.com/InstantID/InstantID) and [InvokeAI](https://github.com/invoke-ai/InvokeAI/releases/tag/v3.6.1).
* **2024-01-25:** Support [video depth visualization](./run_video.py). An [online demo for video](https://huggingface.co/spaces/JohanDL/Depth-Anything-Video) is also available.
* **2024-01-23:** The new ControlNet based on Depth Anything is integrated into [ControlNet WebUI](https://github.com/Mikubill/sd-webui-controlnet) and [ComfyUI's ControlNet](https://github.com/Fannovel16/comfyui_controlnet_aux).
* **2024-01-23:** Depth Anything [ONNX](https://github.com/fabio-sim/Depth-Anything-ONNX) and [TensorRT](https://github.com/spacewalk01/depth-anything-tensorrt) versions are supported.
* **2024-01-22:** Paper, project page, code, models, and demo ([HuggingFace](https://huggingface.co/spaces/LiheYoung/Depth-Anything), [OpenXLab](https://openxlab.org.cn/apps/detail/yyfan/depth_anything)) are released.


## Features of Depth Anything

***If you need other features, please first check [existing community supports](#community-support).***

- **Relative depth estimation**:
    
    Our foundation models listed [here](https://huggingface.co/spaces/LiheYoung/Depth-Anything/tree/main/checkpoints) can provide relative depth estimation for any given image robustly. Please refer [here](#running) for details.

- **Metric depth estimation**

    We fine-tune our Depth Anything model with metric depth information from NYUv2 or KITTI. It offers strong capabilities of both in-domain and zero-shot metric depth estimation. Please refer [here](./metric_depth) for details.


- **Better depth-conditioned ControlNet**

    We re-train **a better depth-conditioned ControlNet** based on Depth Anything. It offers more precise synthesis than the previous MiDaS-based ControlNet. Please refer [here](./controlnet/) for details. You can also use our new ControlNet based on Depth Anything in [ControlNet WebUI](https://github.com/Mikubill/sd-webui-controlnet) or [ComfyUI's ControlNet](https://github.com/Fannovel16/comfyui_controlnet_aux).



You can easily load our pre-trained models by:
```python
from depth_anything.dpt import DepthAnything

encoder = 'vits' # can also be 'vitb' or 'vitl'
depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder))
```

Depth Anything is also supported in [``transformers``](https://github.com/huggingface/transformers). You can use it for depth prediction within [3 lines of code](https://huggingface.co/docs/transformers/main/model_doc/depth_anything) (credit to [@niels](https://huggingface.co/nielsr)).

### *No network connection, cannot load these models?*

<details>
<summary>Click here for solutions</summary>

- First, manually download the three checkpoints: [depth-anything-large](https://huggingface.co/spaces/LiheYoung/Depth-Anything/blob/main/checkpoints/depth_anything_vitl14.pth), [depth-anything-base](https://huggingface.co/spaces/LiheYoung/Depth-Anything/blob/main/checkpoints/depth_anything_vitb14.pth), and [depth-anything-small](https://huggingface.co/spaces/LiheYoung/Depth-Anything/blob/main/checkpoints/depth_anything_vits14.pth).

- Second, upload the folder containing the checkpoints to your remote server.

- Lastly, load the model locally:
```python
from depth_anything.dpt import DepthAnything

model_configs = {
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
}

encoder = 'vitl' # or 'vitb', 'vits'
depth_anything = DepthAnything(model_configs[encoder])
depth_anything.load_state_dict(torch.load(f'./checkpoints/depth_anything_{encoder}14.pth'))
```
Note that in this locally loading manner, you also do not have to install the ``huggingface_hub`` package. In this way, please feel free to delete this [line](https://github.com/LiheYoung/Depth-Anything/blob/e7ef4b4b7a0afd8a05ce9564f04c1e5b68268516/depth_anything/dpt.py#L5) and the ``PyTorchModelHubMixin`` in this [line](https://github.com/LiheYoung/Depth-Anything/blob/e7ef4b4b7a0afd8a05ce9564f04c1e5b68268516/depth_anything/dpt.py#L169).
</details>


## Usage 

### Installation

```bash
git clone https://github.com/LiheYoung/Depth-Anything
cd Depth-Anything
pip install -r requirements.txt
```

### Running

```bash
python run.py --encoder <vits | vitb | vitl> --img-path <img-directory | single-img | txt-file> --outdir <outdir> [--pred-only] [--grayscale]
```
Arguments:
- ``--img-path``: you can either 1) point it to an image directory storing all interested images, 2) point it to a single image, or 3) point it to a text file storing all image paths.
- ``--pred-only`` is set to save the predicted depth map only. Without it, by default, we visualize both image and its depth map side by side.
- ``--grayscale`` is set to save the grayscale depth map. Without it, by default, we apply a color palette to the depth map.

For example:
```bash
python run.py --encoder vitl --img-path assets/examples --outdir depth_vis
```

**If you want to use Depth Anything on videos:**
```bash
python run_video.py --encoder vitl --video-path assets/examples_video --outdir video_depth_vis
```

### Gradio demo <a href='https://github.com/gradio-app/gradio'><img src='https://img.shields.io/github/stars/gradio-app/gradio'></a> 

To use our gradio demo locally:

```bash
python app.py
```

You can also try our [online demo](https://huggingface.co/spaces/LiheYoung/Depth-Anything).

### Import Depth Anything to your project

If you want to use Depth Anything in your own project, you can simply follow [``run.py``](run.py) to load our models and define data pre-processing. 

<details>
<summary>Code snippet (note the difference between our data pre-processing and that of MiDaS)</summary>

```python
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

import cv2
import torch
from torchvision.transforms import Compose

encoder = 'vits' # can also be 'vitb' or 'vitl'
depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder)).eval()

transform = Compose([
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])

image = cv2.cvtColor(cv2.imread('your image path'), cv2.COLOR_BGR2RGB) / 255.0
image = transform({'image': image})['image']
image = torch.from_numpy(image).unsqueeze(0)

# depth shape: 1xHxW
depth = depth_anything(image)
```
</details>

### Do not want to define image pre-processing or download model definition files?

Easily use Depth Anything through [``transformers``](https://github.com/huggingface/transformers) within 3 lines of code! Please refer to [these instructions](https://huggingface.co/docs/transformers/main/model_doc/depth_anything) (credit to [@niels](https://huggingface.co/nielsr)).

**Note:** If you encounter ``KeyError: 'depth_anything'``, please install the latest [``transformers``](https://github.com/huggingface/transformers) from source:
```bash
pip install git+https://github.com/huggingface/transformers.git
```
<details>
<summary>Click here for a brief demo:</summary>

```python
from transformers import pipeline
from PIL import Image

image = Image.open('Your-image-path')
pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")
depth = pipe(image)["depth"]
```
</details>

## Community Support

**We sincerely appreciate all the extensions built on our Depth Anything from the community. Thank you a lot!**

Here we list the extensions we have found:
- Depth Anything TensorRT: 
    - https://github.com/spacewalk01/depth-anything-tensorrt
    - https://github.com/thinvy/DepthAnythingTensorrtDeploy
    - https://github.com/daniel89710/trt-depth-anything
- Depth Anything ONNX: https://github.com/fabio-sim/Depth-Anything-ONNX
- Depth Anything in Transformers.js (3D visualization): https://huggingface.co/spaces/Xenova/depth-anything-web
- Depth Anything for video (online demo): https://huggingface.co/spaces/JohanDL/Depth-Anything-Video
- Depth Anything in ControlNet WebUI: https://github.com/Mikubill/sd-webui-controlnet
- Depth Anything in ComfyUI's ControlNet: https://github.com/Fannovel16/comfyui_controlnet_aux
- Depth Anything in X-AnyLabeling: https://github.com/CVHub520/X-AnyLabeling
- Depth Anything in OpenXLab: https://openxlab.org.cn/apps/detail/yyfan/depth_anything
- Depth Anything in OpenVINO: https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks/280-depth-anything
- Depth Anything ROS:
    - https://github.com/scepter914/DepthAnything-ROS
    - https://github.com/polatztrk/depth_anything_ros
- Depth Anything Android:
    - https://github.com/FeiGeChuanShu/ncnn-android-depth_anything
    - https://github.com/shubham0204/Depth-Anything-Android
- Depth Anything in TouchDesigner: https://github.com/olegchomp/TDDepthAnything
- LearnOpenCV research article on Depth Anything: https://learnopencv.com/depth-anything
     


