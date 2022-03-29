# Copyright (c) OpenMMLab. All rights reserved.
# from asyncio.windows_events import NULL
import warnings

import torch

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector

import onnxruntime
import kp
import cv2
import numpy as np

@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(SingleStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_kneron(self, img, img_metas, rescale=False, return_loss=False):

        for var, name in [(img, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        num_augs = len(img)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(img)}) '
                             f'!= num of image meta ({len(img_metas)})')

        # NOTE the batched image size information may be useful, e.g.
        # in DETR, this is needed for the construction of masks, which is
        # then used for the transformer_head.
        for im, img_meta in zip(img, img_metas):
            batch_size = len(img_meta)
            for img_id in range(batch_size):
                img_meta[img_id]['batch_input_shape'] = tuple(im.size()[-2:])

        outs = None
        if 'proposals' in img_meta:
            img_meta['proposals'] = img_meta['proposals'][0]
        img = img[0]
        img_metas = img_metas[0]

        results_list = None
        if hasattr(self, '__Kn_ONNX_Sess__'):
            tmp = getattr(self, '__Kn_ONNX_Sess__').run(None, {'input': img.cpu().detach().numpy()})
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            res = []

            for o in tmp:
                res.append( torch.from_numpy(o).float().to(device))

            assert hasattr(self.bbox_head, 'get_bboxes_kn'), 'Error: None implemented kneron bbox_head forward type!'

            results_list = self.bbox_head.get_bboxes_kn(
                res, img_metas=img_metas, rescale=rescale)

        elif hasattr(self, '__Kn_PLUS_Params__'):
            tmp_img = img.cpu().detach().numpy()[0].transpose(1,2,0)
            tmp_img *= 256
            tmp_img += 128
            tmp_img = tmp_img.astype(np.uint8)
            img_bgr565 = cv2.cvtColor(src=tmp_img, code=cv2.COLOR_BGR2BGR565)

            kp_params = getattr(self, '__Kn_PLUS_Params__')
            kp.inference.generic_raw_inference_send(device_group=kp_params['device_group'],
                                                    generic_raw_image_header=kp_params['generic_raw_image_header'],
                                                    image=img_bgr565,
                                                    image_format=kp.ImageFormat.KP_IMAGE_FORMAT_RGB565)

            generic_raw_result = kp.inference.generic_raw_inference_receive(device_group=kp_params['device_group'],
                                                                            generic_raw_image_header=kp_params['generic_raw_image_header'],
                                                                            model_nef_descriptor=kp_params['model_nef_descriptor'])
            inf_node_output_list = []
            for node_idx in range(generic_raw_result.header.num_output_node):
                inference_float_node_output = kp.inference.generic_inference_retrieve_float_node(node_idx=node_idx,
                                                                                                generic_raw_result=generic_raw_result,
                                                                                                channels_ordering=kp.ChannelOrdering.KP_CHANNEL_ORDERING_CHW
                                                                                                )
                inf_node_output_list.append(inference_float_node_output.ndarray.copy())
            assert hasattr(self.bbox_head, 'get_bboxes_kn'), 'Error: None implemented kneron bbox_head forward type!'

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            
            res = []
            for o1 in inf_node_output_list:
                res.append( torch.from_numpy(o1).float().to(device))
            results_list = self.bbox_head.get_bboxes_kn(
                res, img_metas=img_metas, rescale=rescale)

        else: # debug purpose
            feat = self.extract_feat(img)
            outs = self.bbox_head(feat)

            results_list = self.bbox_head.get_bboxes(
                *outs, img_metas=img_metas, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]

        return bbox_results

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas, with_nms=True):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape

        if len(outs) == 2:
            # add dummy score_factor
            outs = (*outs, None)
        # TODO Can we change to `get_bboxes` when `onnx_export` fail
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            *outs, img_metas, with_nms=with_nms)

        return det_bboxes, det_labels
