# Kneron AI Training/Deployment Platform (mmDetection-based)


## Introduction

---
  [Kneron-AI_Training_mmDetection](https://github.com/kneron/AI_Training_mmDetection) is a platform built upon the well-known [mmdetection](https://github.com/open-mmlab/mmdetection) for detection and instance segmentation. We encourage that you may start with the [mmdetection docs](https://mmdetection.readthedocs.io/en/latest/) for basics. 

  In this repository, we provide an end-to-end training/deployment flow to realize on Kneron's AI accelerators: 

  1. **Training/Evalulation:**
      - Modified model configuration and verified for Kneron hardware platform 
      - Please see [Overview of Benchmark and Model Zoo](#Overview-of-Benchmark-and-Model-Zoo) for the model list
  2. **Converting to onnx:** 
      - pytorch2onnx_kneron.py (beta)
      - Export *optimized* and *Kneron-toolchain supported* onnx
          - Automatically modify model for arbitrary data normalization preprocess
  3. **General Evaluation**
      - test_kneron.py (beta)
      - Evaluation the model with *pytorch checkpoint, onnx, and kneron-nef*
  4. **Testing**
      - inference_kn (beta)
      - Verify the converted [NEF](http://doc.kneron.com/docs/#toolchain/manual/#5-nef-workflow) model on Kneron USB accelerator with this api
  5. **Converting Kneron-NEF:** (tool-chain feature)
     - Convert the trained pytorch model to [Kneron-NEF](http://doc.kneron.com/docs/#toolchain/manual/#5-nef-workflow) model, which could be used on Kneron hardware platform.

## License
---

This project is released under the [Apache 2.0 license](LICENSE).

## Changelog
---
N/A

## Overview of Benchmark and Model Zoo
---
N/A

## Installation
---
- Please refer to [get_started.md](docs/en/get_started.md) for installation.
- Please refer to [Kneron PLUS - Python: Installation](http://doc.kneron.com/docs/#plus_python/introduction/install_dependency/) for the environment setup for Kneron USB accelerator.

## Getting Started
---
### Tutorial - Kneron Edition
- [YOLOX: Step-By-Step](link/to/yolox-step-by-step.md): A tutorial for users to get started easily. To see detailed documents, please see below.

### Documents - Kneron Edition
- [Kneron ONNX Export](link/to/pytorch2onnx_kneron/doc.md)
- [Kneron Inference](link/to/inference_kneron/doc.md)
- [Kneron toolchain step-by-step (YOLOv3)](http://doc.kneron.com/docs/#toolchain/manual/#0-overview)
- [Kneron toolchain manual](http://doc.kneron.com/docs/#toolchain/manual/#0-overview)

### Original MMDetection Documents
- [original mmdetection getting started](https://github.com/open-mmlab/mmdetection#getting-started): It is recommended to read original MMDetection getting started documents for other MMDetection operations.
- [original mmdetection readthedoc](https://mmdetection.readthedocs.io/en/latest/): Detailed original MMDetection documents.

## Contributing
---
[Kneron-AI_Training_mmDetection](https://github.com/kneron/AI_Training_mmDetection) a platform built upon [OpenMMLab-mmdetection](https://github.com/open-mmlab/mmdetection)

- For issues regarding to [mmdetection](https://github.com/open-mmlab/mmdetection):
We appreciate all contributions to improve [OpenMMLab-mmdetection](https://github.com/open-mmlab/mmdetection). Ongoing projects can be found in out [GitHub Projects](https://github.com/open-mmlab/mmdetection/projects). Welcome community users to participate in these projects. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.

- For issues regarding to this repository [Kneron-AI_Training_mmDetection](https://github.com/kneron/AI_Training_mmDetection): Welcome leave the comment or submit the pull request here to improve Kneron-AI_Training_mmDetection


## Related Projects
- MMSeg_Kneron: Kneron training/deployment platform on [OpenMMLab -mmSegmentation](https://github.com/open-mmlab/mmsegmentation) semantic segmentation toolbox
