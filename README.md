# Kneron AI Training/Deployment Platform (mmDetection-based)

## Introduction

  [kneron-mmdetection](https://github.com/kneron/kneron-mmdetection) is a platform built upon the well-known [mmdetection](https://github.com/open-mmlab/mmdetection) for detection and instance segmentation. We encourage you to start with [YOLOX: Step-By-Step](docs_kneron/yolox_step_by_step.md) to build basic knowledge of Kneron-Edition mmdetection, and read [mmdetection docs](https://mmdetection.readthedocs.io/en/latest/) for detailed mmdetection usage.

  In this repository, we provide an end-to-end training/deployment flow to realize on Kneron's AI accelerators: 

  1. **Training/Evalulation:**
      - Modified model configuration file and verified for Kneron hardware platform 
      - Please see [Overview of Benchmark and Model Zoo](#Overview-of-Benchmark-and-Model-Zoo) for Kneron-Verified model list
  2. **Converting to ONNX:** 
      - pytorch2onnx_kneron.py (beta)
      - Export *optimized* and *Kneron-toolchain supported* onnx
          - Automatically modify model for arbitrary data normalization preprocess
  3. **Evaluation**
      - test_kneron.py (beta)
      - Evaluate the model with *pytorch checkpoint, onnx, and kneron-nef*
  4. **Testing**
      - inference_kn (beta)
      - Verify the converted [NEF](http://doc.kneron.com/docs/#toolchain/manual/#5-nef-workflow) model on Kneron USB accelerator with this API
  5. **Converting Kneron-NEF:** (toolchain feature)
     - Convert the trained pytorch model to [Kneron-NEF](http://doc.kneron.com/docs/#toolchain/manual/#5-nef-workflow) model, which could be used on Kneron hardware platform.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Changelog
N/A

## Overview of Benchmark and Kneron Model Zoo
| Backbone  | size   | Mem (GB) |   box AP | Config | Download |
|:---------:|:-------:|:-------:|:-------:|:--------:|:------:|
| YOLOX-s | 640 |   7.6      |   40.5  | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/yolox/yolox_s_8x8_300e_coco.py)       |[model](https://github.com/kneron/Model_Zoo/blob/main/mmdetection/yolox_s/latest.zip)

## Installation
- Please refer to [YOLOX: Step-By-Step, Step 0. Environment](docs_kneron/yolox_step_by_step.md) for installation.
- Please refer to [Kneron PLUS - Python: Installation](http://doc.kneron.com/docs/#plus_python/introduction/install_dependency/) for the environment setup for Kneron USB accelerator.

## Getting Started
### Tutorial - Kneron Edition
- [YOLOX: Step-By-Step](docs_kneron/yolox_step_by_step.md): A tutorial for users to get started easily. To see detailed documents, please see below.

### Documents - Kneron Edition
- [Kneron ONNX Export] (under development)
- [Kneron Inference] (under development)
- [Kneron Toolchain Step-By-Step (YOLOv3)](http://doc.kneron.com/docs/#toolchain/yolo_example/)
- [Kneron Toolchain Manual](http://doc.kneron.com/docs/#toolchain/manual/#0-overview)

### Original mmdetection Documents
- [Original mmdetection getting started](https://github.com/open-mmlab/mmdetection#getting-started): It is recommended to read the original mmdetection getting started documents for other mmdetection operations.
- [Original mmdetection readthedoc](https://mmdetection.readthedocs.io/en/latest/): Original mmdetection documents.

## Contributing
[kneron-mmdetection](https://github.com/kneron/kneron-mmdetection) a platform built upon [OpenMMLab-mmdetection](https://github.com/open-mmlab/mmdetection)

- For issues regarding to the original [mmdetection](https://github.com/open-mmlab/mmdetection):
We appreciate all contributions to improve [OpenMMLab-mmdetection](https://github.com/open-mmlab/mmdetection). Ongoing projects can be found in out [GitHub Projects](https://github.com/open-mmlab/mmdetection/projects). Welcome community users to participate in these projects. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

- For issues regarding to this repository [kneron-mmdetection](https://github.com/kneron/kneron-mmdetection): Welcome to leave the comment or submit pull requests here to improve kneron-mmdetection

## Related Projects
- kneron-mmsegmentation: Kneron training/deployment platform on [OpenMMLab - mmSegmentation](https://github.com/open-mmlab/kneron-mmsegmentation) semantic segmentation toolbox
