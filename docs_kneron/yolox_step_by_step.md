# Step 0. Environment

## Prerequisites

- Linux or macOS (Windows is in experimental support)
- Python 3.6+
- PyTorch 1.3+
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)

**Note:** You need to run `pip uninstall mmcv` first if you have mmcv installed.
If mmcv and mmcv-full are both installed, there will be `ModuleNotFoundError`.


### Install MMDetection 

1. Install mmcv-full, we recommend you to install the pre-build package as below.

    ```shell
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
    ```

    Please replace `{cu_version}` and `{torch_version}` in the url to your desired one. For example, to install the `mmcv-full` with `CUDA 10.1` and `PyTorch 1.6.0`, use the following command:

    ```shell
    pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
    ```

    See [here](https://github.com/open-mmlab/mmcv#install-with-pip) for different versions of MMCV compatible to different PyTorch and CUDA versions.

2. Clone the Kneron version mmDetection repository.

    ```bash
    git clone https://github.com/kneron/AI_Training_mmDetection.git
    cd AI_Training_mmDetection
    ```

3. Install build requirements and then install MMDetection.

    ```shell
    pip install -r requirements/build.txt
    pip install -v -e .  # or "python setup.py develop"
    ```

# Step 1: Training models on standard datasets 

MMDetection provides hundreds of existing and existing detection models in [Model Zoo](https://mmdetection.readthedocs.io/en/latest/model_zoo.html)), and supports multiple standard datasets, including Pascal VOC, COCO, CityScapes, LVIS, etc. This note will show how to perform common tasks on these existing models and standard datasets, including:

- Use existing models to inference on given images.
- Test existing models on standard datasets.
- Train models on standard datasets.

## Train models on standard datasets

MMDetection also provides out-of-the-box tools for training detection models.
This section will show how to train models (under [configs](https://github.com/open-mmlab/mmdetection/tree/master/configs)) on standard datasets i.e. COCO.

**Important**: You might need to modify the [config](https://github.com/open-mmlab/mmdetection/blob/5e246d5e3bc3310b5c625fb57bc03d2338ca39bc/docs/en/tutorials/config.md) according your GPUs resource (such as "samples_per_gpu","workers_per_gpu" ...etc due to your GPUs RAM limitation).
The default learning rate in config files is for 8 GPUs and 2 img/gpu (batch size = 8\*2 = 16).
According to the [linear scaling rule](https://arxiv.org/abs/1706.02677), you need to set the learning rate proportional to the batch size if you use different GPUs or images per GPU, e.g., `lr=0.01` for 4 GPUs \* 2 imgs/gpu and `lr=0.08` for 16 GPUs \* 4 imgs/gpu.

### Step 1-1: Prepare datasets

Public datasets like [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/index.html) or mirror and [COCO](https://cocodataset.org/#download) are available from official websites or mirrors. Note: In the detection task, Pascal VOC 2012 is an extension of Pascal VOC 2007 without overlap, and we usually use them together.
It is recommended to download and extract the dataset somewhere outside the project directory and symlink the dataset root to `$MMDETECTION/data` as below.
If your folder structure is different, you may need to change the corresponding paths in config files.

```plain
mmdetection
├── mmdet
├── tools
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
│   ├── VOCdevkit
│   │   ├── VOC2007
│   │   ├── VOC2012
```


### Step 1-2: Training Example with YOLOX:

[YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/abs/2107.08430)


The training of YOLOX is only need to use the configuration file (The configuration is modified to fit Kneron platform spec.):
```python
python tools/train.py /configs/yolox/yolox_s_8x8_300e_coco_img_norm.py
```
* (Note) you might need to create a folder name 'work_dir' in MMDetection root folder because we set 'work_dir' as default folder in 'yolox_s_8x8_300e_coco_img_norm.py'
* (Note 2) the whole training process takes very long time, if you just want to have a quick look the all flow, we recommend you can download our trained model and skip this process
```bash
mkdir work_dirs
cd work_dirs
wget https://github.com/kneron/Model_Zoo/raw/main/mmdetection/yolox_s/latest.zip
unzip latest.zip
cd ..
```
* (Note 3) this is a "from scratch training" tutorial, and might need lot's of time and gpu resource. If you want to train a model to detect specific object, recommend you can read the [finetune.md](https://github.com/open-mmlab/mmdetection/blob/5e246d5e3bc3310b5c625fb57bc03d2338ca39bc/docs/en/tutorials/finetune.md) and [customize_dataset.md](https://github.com/open-mmlab/mmdetection/blob/5e246d5e3bc3310b5c625fb57bc03d2338ca39bc/docs/en/tutorials/customize_dataset.md)

# Step 2: Test trained model
'tools/test_kneron.py' is a script which generates inference result and (if `--eval` given) evaluate the results to see if our pytorch model is well trained. It's always good to evluate our pytorch model before deploying it.

```python
python tools/test_kneron.py \
    configs/yolox/yolox_s_8x8_300e_coco_img_norm.py \
    work_dirs/latest.pth \
    --eval bbox \
    --out-kneron output.json
```
* 'configs/yolox/yolox_s_8x8_300e_coco_img_norm.py' is your yolox training config
* 'work_dirs/latest.pth' is your trained yolox model

The expected result of the command above will be something similar to the following text (the numbers may slightly differ):
```
...
Average Precision (AP) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.379
Average Precision (AP) @[ IoU=0.50 | area= all | maxDets=1000 ] = 0.564
Average Precision (AP) @[ IoU=0.75 | area= all | maxDets=1000 ] = 0.410
Average Precision (AP) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.205
Average Precision (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.416
Average Precision (AP) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.503
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets=100 ] = 0.530
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets=300 ] = 0.531
Average Recall (AR) @[ IoU=0.50:0.95 | area= all | maxDets=1000 ] = 0.531
Average Recall (AR) @[ IoU=0.50:0.95 | area= small | maxDets=1000 ] = 0.317
Average Recall (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=1000 ] = 0.582
Average Recall (AR) @[ IoU=0.50:0.95 | area= large | maxDets=1000 ] = 0.678

OrderedDict([('bbox_mAP', 0.379), ('bbox_mAP_50', 0.564), ('bbox_mAP_75', 0.41), ('bbox_mAP_s', 0.205), ('bbox_mAP_m', 0.416), ('bbox_mAP_l', 0.503), ('bbox_mAP_copypaste', '0.379 0.564 0.410 0.205 0.416 0.503')])
...
```

# Step 3: Export onnx
'tools/deployment/pytorch2onnx.py' is a script provided by MMDetection to help user to convert our trained pth model to onnx:
```python
python tools/deployment/pytorch2onnx.py \
    configs/yolox/yolox_s_8x8_300e_coco_img_norm.py \
    work_dirs/yolox_s_8x8_300e_coco_img_norm/latest.pth \
    --output-file work_dirs/latest.onnx \
    --skip-postprocess \
    --shape 640 640
```
* 'configs/yolox/yolox_s_8x8_300e_coco_img_norm.py' is your yolox training config
* 'work_dirs/latest.pth' is your trained yolox model

The output onnx should be the same name as 'work_dirs/latest.pth' with '.onnx' post-fix in the same folder.


# Step 4: Convert onnx to [NEF](http://doc.kneron.com/docs/#toolchain/manual/#5-nef-workflow) model for Kneron platform
 
### Step 4-1: Install Kneron toolchain docker:
* check [document](http://doc.kneron.com/docs/#toolchain/manual/#1-installation)

### Step 4-2: Mout Kneron toolchain docker 
* Mount a folder (e.g. '/mnt/hgfs/Competition') to toolchain docker as '/data1', the converted onnx in Step 3 should be put here, all the toolchain operation should happen in this folder.
```
sudo docker run --rm -it -v /mnt/hgfs/Competition:/data1 kneron/toolchain:latest
```

### Step 4-3: Import KTC and required lib in python shell
* Now, we go through all toolchain flow by KTC (Kneron Toolchain) using the Python API in the Python shell
```python
import ktc
import numpy as np
import os
import onnx
from PIL import Image
```

### Step 4-4: optimize the onnx model
```python
onnx_path = '/data1/latest.onnx'
m = onnx.load(onnx_path)
m = ktc.onnx_optimizer.onnx2onnx_flow(m)
onnx.save(m,'latest.opt.onnx')
```

### Step 4-5: config and load nessasary data for ktc, and check onnx is ok for toolchain
```python 
# npu (only) performance simulation
km = ktc.ModelConfig(20008, "0001", "720", onnx_model=m)
eval_result = km.evaluate()
print("\nNpu performance evaluation result:\n" + str(eval_result))
```

### Step 4-6: quantize the onnx model
We use [random picked voc dataset](https://www.kneron.com/forum/uploads/112/SMZ3HLBK3DXJ.7z) (50 images) as quantization data , we have to
1. download the data 
2. uncompression the data as folder named "voc_data50" 
3. put the "voc_data50" into docker mounted folder (in docker, the path looks like "/data1/voc_data50")

the following script will do some preprocess(should be the same as training code) on our quantization data, and put it in a list:
```python
import os
from os import walk

img_list = []
for (dirpath, dirnames, filenames) in walk("/data1/voc_data50"):
    for f in filenames:
        fullpath = os.path.join(dirpath, f)
        
        image = Image.open(fullpath)
        image = image.convert("RGB")
        image = Image.fromarray(np.array(image)[...,::-1])
        img_data = np.array(image.resize((640, 640), Image.BILINEAR)) / 256 - 0.5
        print(fullpath)
        img_list.append(img_data)
```

Then, perform quantization. The BIE model will be generated at /data1/output.bie.

```python
# fix point analysis
bie_model_path = km.analysis({"input": img_list})
print("\nFix point analysis done. Save bie model to '" + str(bie_model_path) + "'")
```

### Step 4-7: Compile
The final step is compile the BIE model into an NEF model.
```python
# compile
nef_model_path = ktc.compile([km])
print("\nCompile done. Save Nef file to '" + str(nef_model_path) + "'")
```

You can find the NEF file under /data1/batch_compile/models_720.nef. models_720.nef is the final compiled model.

# Step 5: Run [NEF](http://doc.kneron.com/docs/#toolchain/manual/#5-nef-workflow) model on KL720

* Check Kneron PLUS official document:
  * python version:
    http://doc.kneron.com/docs/#plus_python/#_top
  * C version:
    http://doc.kneron.com/docs/#plus_c/getting_started/


