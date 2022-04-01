from mmdet.apis import init_detector, inference_detector, show_result_pyplot
import onnxruntime
import cv2
import numpy as np
import argparse
#from mmdet.core import ( build_anchor_generator)
import math
import mmcv
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMDetection models to ONNX')
    parser.add_argument('--config-file', help='test config file path', default='./configs/yolof/yolof_r50_c5_8x8_1x_coco_kneron_norm_torch_nograd_noeval.py')
    parser.add_argument('--model-cfg', help='the model configuration dictionary', default='./model_config/yolof_model_cfg.json')
    parser.add_argument('--onnx-file', help='onnx file', default='./checkpoints/yolof_test_0707_model_only_kneron_optimized.onnx')
    parser.add_argument('--input-img', type=str, help='Images for input', default= './demo/sample.jpg')
    parser.add_argument('--output-img', type=str, help='Images for output', default= './demo/sample_onnx_yolof_0720.jpg')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[384, 640],
        help='input image size')
    args = parser.parse_args()
    return args

COCO_CLASSES = ('background','person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
           'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
           'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
           'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
           'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')




class AnchorGenerator:
    """Standard anchor generator for 2D anchor-based detectors.

    Args:
        strides (list[int] | list[tuple[int, int]]): Strides of anchors
            in multiple feature levels in order (w, h).
        ratios (list[float]): The list of ratios between the height and width
            of anchors in a single level.
        scales (list[int] | None): Anchor scales for anchors in a single level.
            It cannot be set at the same time if `octave_base_scale` and
            `scales_per_octave` are set.
        base_sizes (list[int] | None): The basic sizes
            of anchors in multiple levels.
            If None is given, strides will be used as base_sizes.
            (If strides are non square, the shortest stride is taken.)
        scale_major (bool): Whether to multiply scales first when generating
            base anchors. If true, the anchors in the same row will have the
            same scales. By default it is True in V2.0
        octave_base_scale (int): The base scale of octave.
        scales_per_octave (int): Number of scales for each octave.
            `octave_base_scale` and `scales_per_octave` are usually used in
            retinanet and the `scales` should be None when they are set.
        centers (list[tuple[float, float]] | None): The centers of the anchor
            relative to the feature grid center in multiple feature levels.
            By default it is set to be None and not used. If a list of tuple of
            float is given, they will be used to shift the centers of anchors.
        center_offset (float): The offset of center in proportion to anchors'
            width and height. By default it is 0 in V2.0.

    Examples:
        >>> from mmdet.core import AnchorGenerator
        >>> self = AnchorGenerator([16], [1.], [1.], [9])
        >>> all_anchors = self.grid_anchors([(2, 2)], device='cpu')
        >>> print(all_anchors)
        [tensor([[-4.5000, -4.5000,  4.5000,  4.5000],
                [11.5000, -4.5000, 20.5000,  4.5000],
                [-4.5000, 11.5000,  4.5000, 20.5000],
                [11.5000, 11.5000, 20.5000, 20.5000]])]
        >>> self = AnchorGenerator([16, 32], [1.], [1.], [9, 18])
        >>> all_anchors = self.grid_anchors([(2, 2), (1, 1)], device='cpu')
        >>> print(all_anchors)
        [tensor([[-4.5000, -4.5000,  4.5000,  4.5000],
                [11.5000, -4.5000, 20.5000,  4.5000],
                [-4.5000, 11.5000,  4.5000, 20.5000],
                [11.5000, 11.5000, 20.5000, 20.5000]]), \
        tensor([[-9., -9., 9., 9.]])]
    """

    def __init__(self,
                 strides,
                 ratios,
                 scales=None,
                 base_sizes=None,
                 scale_major=True,
                 octave_base_scale=None,
                 scales_per_octave=None,
                 centers=None,
                 center_offset=0.):
        # check center and center_offset

        if center_offset != 0:
            assert centers is None, 'center cannot be set when center_offset' \
                f'!=0, {centers} is given.'
        if not (0 <= center_offset <= 1):
            raise ValueError('center_offset should be in range [0, 1], '
                             f'{center_offset} is given.')
        if centers is not None:
            assert len(centers) == len(strides), \
                'The number of strides should be the same as centers, got ' \
                f'{strides} and {centers}'

        # calculate base sizes of anchors
        self.strides = [(stride,stride) for stride in strides]
        self.base_sizes = [min(stride) for stride in self.strides
                           ] if base_sizes is None else base_sizes
        assert len(self.base_sizes) == len(self.strides), \
            'The number of strides should be the same as base sizes, got ' \
            f'{self.strides} and {self.base_sizes}'
        # calculate scales of anchors
        assert ((octave_base_scale is not None
                and scales_per_octave is not None) ^ (scales is not None)), \
            'scales and octave_base_scale with scales_per_octave cannot' \
            ' be set at the same time'
        if scales is not None:
            self.scales = np.array(scales) #scales=[1, 2, 4, 8, 16]
        elif octave_base_scale is not None and scales_per_octave is not None:
            octave_scales = np.array(
                [2**(i / scales_per_octave) for i in range(scales_per_octave)])
            scales = octave_scales * octave_base_scale
            self.scales = np.array(scales)
        else:
            raise ValueError('Either scales or octave_base_scale with '
                             'scales_per_octave should be set')

        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.ratios = np.array(ratios) #ratios=[1.0]
        self.scale_major = scale_major
        self.centers = centers
        self.center_offset = center_offset
        self.base_anchors = self.gen_base_anchors()

    def gen_base_anchors(self):
        """Generate base anchors.

        Returns:
            list(np.array): Base anchors of a feature grid in multiple \
                feature levels.
        """
        multi_level_base_anchors = []
        #print("self.base_sizes:", self.base_sizes) #[32]
        for i, base_size in enumerate(self.base_sizes):
            center = None
            if self.centers is not None:
                center = self.centers[i]
            multi_level_base_anchors.append(
                self.gen_single_level_base_anchors(
                    base_size,
                    scales=self.scales,
                    ratios=self.ratios,
                    center=center))
        return multi_level_base_anchors

    def gen_single_level_base_anchors(self,
                                      base_size,
                                      scales,
                                      ratios,
                                      center=None):
        """Generate base anchors of a single level.

        Args:
            base_size (int | float): Basic size of an anchor.
            scales (np.array): Scales of the anchor.
            ratios (np.array): The ratio between between the height
                and width of anchors in a single level.
            center (tuple[float], optional): The center of the base anchor
                related to a single feature grid. Defaults to None.

        Returns:
            torch.Tensor: Anchors in a single-level feature maps.
        """
        w = base_size
        h = base_size
        if center is None:
            x_center = self.center_offset * w
            y_center = self.center_offset * h
        else:
            x_center, y_center = center

        h_ratios = np.sqrt(ratios)
        w_ratios = 1 / h_ratios
        #scale = [1, 2, 4, 8, 16]
        if self.scale_major:
            ws = (w * w_ratios[:, None] * scales[None, :]).reshape(-1)
            hs = (h * h_ratios[:, None] * scales[None, :]).reshape(-1)
        else:
            ws = (w * scales[:, None] * w_ratios[None, :]).reshape(-1)
            hs = (h * scales[:, None] * h_ratios[None, :]).reshape(-1)

        # use float anchor and the anchor's center is aligned with the
        # pixel center
        base_anchors = [
            x_center - 0.5 * ws, y_center - 0.5 * hs, x_center + 0.5 * ws,
            y_center + 0.5 * hs
        ]

        #base_anchors = torch.stack(base_anchors, dim=-1)
        base_anchors = np.stack(base_anchors, axis=-1) # (5,4)
        return base_anchors

    @property
    def num_base_anchors(self):
        """list[int]: total number of base anchors in a feature grid"""
        return [base_anchors.size(0) for base_anchors in self.base_anchors]

    @property
    def num_levels(self):
        """int: number of feature levels that the generator will be applied"""
        return len(self.strides)

    def grid_anchors(self, featmap_sizes):
        """Generate grid anchors in multiple feature levels.

        Args:
            featmap_sizes (list[tuple]): List of feature map sizes in
                multiple feature levels.
            device (str): Device where the anchors will be put on.

        Return:
            list[torch.Tensor]: Anchors in multiple feature levels. \
                The sizes of each tensor should be [N, 4], where \
                N = width * height * num_base_anchors, width and height \
                are the sizes of the corresponding feature level, \
                num_base_anchors is the number of anchors for that level.
        """

        assert self.num_levels == len(featmap_sizes)
        multi_level_anchors = []
        for i in range(self.num_levels):
            anchors = self.single_level_grid_anchors(
                self.base_anchors[i],
                featmap_sizes[i],
                self.strides[i])
            multi_level_anchors.append(anchors)
        return multi_level_anchors

    def single_level_grid_anchors(self,
                                  base_anchors,
                                  featmap_size,
                                  stride=(16, 16)):
        """Generate grid anchors of a single level.

        Note:
            This function is usually called by method ``self.grid_anchors``.

        Args:
            base_anchors (torch.Tensor): The base anchors of a feature grid.
            featmap_size (tuple[int]): Size of the feature maps.
            stride (tuple[int], optional): Stride of the feature map in order
                (w, h). Defaults to (16, 16).
            device (str, optional): Device the tensor will be put on.
                Defaults to 'cuda'.

        Returns:
            torch.Tensor: Anchors in the overall feature maps.
        """
        # keep as Tensor, so that we can covert to ONNX correctly
        feat_h, feat_w = featmap_size
        shift_x = np.arange(0, feat_w) * stride[0]
        shift_y = np.arange(0, feat_h) * stride[1]

        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        #shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts = np.stack([shift_xx, shift_yy, shift_xx, shift_yy], axis=-1)

        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)
        #print(base_anchors.shape) # (5,4)
        #print(shifts.shape) # (240,4)
        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.reshape(-1, 4) #(1200,4)
        #print(all_anchors.shape) #
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors

    def _meshgrid(self, x, y, row_major=True):
        """Generate mesh grid of x and y.

        Args:
            x (np.array): Grids of x dimension.
            y (np.array): Grids of y dimension.
            row_major (bool, optional): Whether to return y grids first.
                Defaults to True.

        Returns:
            tuple[np.array]: The mesh grids of x and y.
        """
        # use shape instead of len to keep tracing while exporting to onnx
        xx = np.tile(x, y.shape[0])
        y = y.reshape(-1, 1)
        y = np.tile(y,(1, x.shape[0]))
        yy = y.reshape(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx




def area_of(left_top, right_bottom):
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = np.clip(right_bottom - left_top, a_min=0.0, a_max=None)
    return hw[..., 0] * hw[..., 1]

def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.
    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])


    return overlap_area / (area0 + area1 - overlap_area + eps)

def hard_nms(box_scores, iou_threshold=0.5, top_k=-1, candidate_size=200):
    """

    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []

    indexes = np.argsort(scores)[::-1]

    indexes = indexes[:candidate_size]
    while len(indexes) > 0:
        current = indexes[0]
        picked.append(current.item())
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, 0)
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


def delta2bbox(rois,
               deltas,
               means=(0., 0., 0., 0.),
               stds=(1., 1., 1., 1.),
               max_shape=None,
               wh_ratio_clip=16 / 1000,
               clip_border=True,
               add_ctr_clamp=False,
               ctr_clamp=32):
    """Apply deltas to shift/scale base boxes.

    Typically the rois are anchor or proposed bounding boxes and the deltas are
    network outputs used to shift/scale those boxes.
    This is the inverse function of :func:`bbox2delta`.

    Args:
        rois (Tensor): Boxes to be transformed. Has shape (N, 4) or (B, N, 4)
        deltas (Tensor): Encoded offsets with respect to each roi.
            Has shape (B, N, num_classes * 4) or (B, N, 4) or
            (N, num_classes * 4) or (N, 4). Note N = num_anchors * W * H
            when rois is a grid of anchors.Offset encoding follows [1]_.
        means (Sequence[float]): Denormalizing means for delta coordinates
        stds (Sequence[float]): Denormalizing standard deviation for delta
            coordinates
        max_shape (Sequence[int] or torch.Tensor or Sequence[
            Sequence[int]],optional): Maximum bounds for boxes, specifies
            (H, W, C) or (H, W). If rois shape is (B, N, 4), then
            the max_shape should be a Sequence[Sequence[int]]
            and the length of max_shape should also be B.
        wh_ratio_clip (float): Maximum aspect ratio for boxes.
        clip_border (bool, optional): Whether clip the objects outside the
            border of the image. Defaults to True.
        add_ctr_clamp (bool): Whether to add center clamp, when added, the
            predicted box is clamped is its center is too far away from
            the original anchor's center. Only used by YOLOF. Default False.
        ctr_clamp (int): the maximum pixel shift to clamp. Only used by YOLOF.
            Default 32.

    Returns:
        Tensor: Boxes with shape (B, N, num_classes * 4) or (B, N, 4) or
           (N, num_classes * 4) or (N, 4), where 4 represent
           tl_x, tl_y, br_x, br_y.

    References:
        .. [1] https://arxiv.org/abs/1311.2524

    Example:
        >>> rois = torch.Tensor([[ 0.,  0.,  1.,  1.],
        >>>                      [ 0.,  0.,  1.,  1.],
        >>>                      [ 0.,  0.,  1.,  1.],
        >>>                      [ 5.,  5.,  5.,  5.]])
        >>> deltas = torch.Tensor([[  0.,   0.,   0.,   0.],
        >>>                        [  1.,   1.,   1.,   1.],
        >>>                        [  0.,   0.,   2.,  -1.],
        >>>                        [ 0.7, -1.9, -0.5,  0.3]])
        >>> delta2bbox(rois, deltas, max_shape=(32, 32, 3))
        tensor([[0.0000, 0.0000, 1.0000, 1.0000],
                [0.1409, 0.1409, 2.8591, 2.8591],
                [0.0000, 0.3161, 4.1945, 0.6839],
                [5.0000, 5.0000, 5.0000, 5.0000]])
    """
    #means = deltas.new_tensor(means).view(1,
    #                                      -1).repeat(1,
    #                                                 deltas.size(-1) // 4)
    #stds = deltas.new_tensor(stds).view(1, -1).repeat(1, deltas.size(-1) // 4)
    #denorm_deltas = deltas * stds + means
    dx = deltas[..., 0::4] #(1, 1200, 1)
    dy = deltas[..., 1::4]
    dw = deltas[..., 2::4]
    dh = deltas[..., 3::4]
    #print("rois.shape: ", rois.shape )  #(1, 1200, 4)
    #print("deltas.shape: ", deltas.shape ) #(1, 1200, 4)

    x1, y1 = rois[..., 0], rois[..., 1] #(1, 1200)
    x2, y2 = rois[..., 2], rois[..., 3]

    # Compute center of each roi
    px = ((x1 + x2) * 0.5)# (1, 1200)
    py = ((y1 + y2) * 0.5)
    # Compute width/height of each roi
    pw = (x2 - x1)
    ph = (y2 - y1)


    px = np.expand_dims(px, axis=2) #(1, 1200, 1)
    py = np.expand_dims(py, axis=2)
    pw = np.expand_dims(pw, axis=2)
    ph = np.expand_dims(ph, axis=2)
    dx_width = pw * dx
    dy_height = ph * dy


    max_ratio = np.abs(np.log(wh_ratio_clip)) #4.135166556742356
    #px, py, pw, ph are anchors
    #exit(0)
    if add_ctr_clamp:
        dx_width = np.clip(dx_width, a_min= -ctr_clamp, a_max= ctr_clamp)
        dy_height = np.clip(dy_height, a_min= -ctr_clamp, a_max= ctr_clamp)
        dw = np.clip(dw, a_min= None, a_max=max_ratio)
        dh = np.clip(dh, a_min= None, a_max=max_ratio)
    else:
        dw = np.clip(dw, a_min= -max_ratio, a_max=max_ratio)
        dh = np.clip(dh, a_min= -max_ratio, a_max=max_ratio)
    # Use exp(network energy) to enlarge/shrink each roi
    gw = pw * np.exp(dw)
    gh = ph * np.exp(dh)
    # Use network energy to shift the center of each roi
    gx = px + dx_width
    gy = py + dy_height
    # Convert center-xy/width/height to top-left, bottom-right
    x1 = gx - gw * 0.5 #(1,200,1)
    y1 = gy - gh * 0.5
    x2 = gx + gw * 0.5
    y2 = gy + gh * 0.5

    x1 = np.clip(x1, 0, max_shape[1])
    y1 = np.clip(y1, 0, max_shape[0])
    x2 = np.clip(x2, 0, max_shape[1])
    y2 = np.clip(y2, 0, max_shape[0])

    bboxes = np.concatenate((x1, y1, x2, y2), axis=-1)

    return bboxes

def decode(bboxes,
           pred_bboxes,
           means=(0., 0., 0., 0.),
           stds=(1., 1., 1., 1.),
           max_shape=None,
           wh_ratio_clip=16 / 1000,
           clip_border=True,
           add_ctr_clamp=True,
           ctr_clamp=32
           ):
    """Apply transformation `pred_bboxes` to `boxes`.

    Args:
        bboxes (torch.Tensor): Basic boxes. Shape (B, N, 4) or (N, 4)
        pred_bboxes (Tensor): Encoded offsets with respect to each roi.
           Has shape (B, N, num_classes * 4) or (B, N, 4) or
           (N, num_classes * 4) or (N, 4). Note N = num_anchors * W * H
           when rois is a grid of anchors.Offset encoding follows [1]_.
        max_shape (Sequence[int] or torch.Tensor or Sequence[
           Sequence[int]],optional): Maximum bounds for boxes, specifies
           (H, W, C) or (H, W). If bboxes shape is (B, N, 4), then
           the max_shape should be a Sequence[Sequence[int]]
           and the length of max_shape should also be B.
        wh_ratio_clip (float, optional): The allowed ratio between
            width and height.

    Returns:
        torch.Tensor: Decoded boxes.
    """

    #assert pred_bboxes.size(0) == bboxes.size(0)
    #if pred_bboxes.ndim == 3:
    #    assert pred_bboxes.size(1) == bboxes.size(1)
    decoded_bboxes = delta2bbox(bboxes, pred_bboxes, means, stds, max_shape, wh_ratio_clip, clip_border,
                                add_ctr_clamp, ctr_clamp)

    return decoded_bboxes

bbox_coder_cfg = dict(
    type='DeltaXYWHBBoxCoder',
    target_means=[.0, .0, .0, .0],
    target_stds=[1., 1., 1., 1.],
    add_ctr_clamp=True,
    ctr_clamp=32)
#yolof_bbox_coder = DeltaXYWHBBoxCoder()

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

def _get_bboxes(mlvl_cls_scores,
                mlvl_bbox_preds,
                mlvl_anchors,
                input_shapes,
                num_classes = 80,
                top_k_num =200,
                prob_threshold = 0.3 ):
    """Transform outputs for a batch item into bbox predictions.

    Args:
        mlvl_cls_scores (list[Tensor]): Each element in the list is
            the scores of bboxes of single level in the feature pyramid,
            has shape (N, num_anchors * num_classes, H, W).
        mlvl_bbox_preds (list[Tensor]):  Each element in the list is the
            bboxes predictions of single level in the feature pyramid,
            has shape (N, num_anchors * 4, H, W).
        mlvl_anchors (list[Tensor]): Each element in the list is
            the anchors of single level in feature pyramid, has shape
            (num_anchors, 4).
        input_shapes (list[tuple[int]]): Each tuple in the list represent
            the shape(height, width, 3) of single image in the batch.
        scale_factors (list[ndarray]): Scale factor of the batch
            image arange as list[(w_scale, h_scale, w_scale, h_scale)].
        cfg (mmcv.Config): Test / postprocessing configuration,
            if None, test_cfg would be used.
        rescale (bool): If True, return boxes in original image space.
            Default: False.
        with_nms (bool): If True, do nms before return boxes.
            Default: True.

    Returns:
        list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
            The first item is an (n, 5) tensor, where 5 represent
            (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
            The shape of the second tensor in the tuple is (n,), and
            each element represents the class label of the corresponding
            box.
    """

    assert len(mlvl_cls_scores) == len(mlvl_bbox_preds) == len(
        mlvl_anchors)
    batch_size = mlvl_cls_scores[0].shape[0]

    #mlvl_bboxes = []
    #mlvl_scores = []
    layer = 0
    for cls_score, bbox_pred, anchors in zip(mlvl_cls_scores,
                                             mlvl_bbox_preds,
                                             mlvl_anchors):
        #print("mlvl_cls_scores[0].shape: ", mlvl_cls_scores[0].shape) #(1, 400, 12, 20)
        #print("mlvl_bbox_preds[0].shape: ", mlvl_bbox_preds[0].shape) #(1, 20, 12, 20)

        cls_score = np.transpose(cls_score, (0,2,3,1)) #(1,  12, 20, 400)
        bbox_pred = np.transpose(bbox_pred, (0,2,3,1)) #(1, 12, 20, 20)
        cls_score = cls_score.reshape(batch_size, -1, num_classes) #(1,  1200, 80)
        bbox_pred = bbox_pred.reshape(batch_size, -1, 4) ##(1,  1200, 4)

        scores = sigmoid(cls_score) #(1,1200,80)

        # anchors = (N , 4)
        # N = width * height * num_base_anchors = 20*12*5
        #anchors = anchors.detach().numpy()
        anchors = anchors.reshape(bbox_pred.shape) #anchors.shape:  torch.Size([1, 1200, 4])

        max_scores = np.max(scores, axis=2) #(1, 1200)
        #_, topk_inds = max_scores.topk(top_k_num)
        topk_inds = np.argsort(max_scores[0])[::-1]
        bbox_pred = bbox_pred[:,topk_inds,:]
        scores = scores[:,topk_inds,:]
        anchors = anchors[:,topk_inds,:]

        bbox_pred = bbox_pred[:,:top_k_num,:]
        scores = scores[:,:top_k_num,:]
        anchors = anchors[:,:top_k_num,:]


        bboxes = decode(anchors, bbox_pred, max_shape=input_shapes)
        if layer == 0:
            batch_mlvl_bboxes = bboxes
            batch_mlvl_scores = scores
        else:
            batch_mlvl_bboxes = np.concatenate((mlvl_bboxes, bboxes), axis=1)
            batch_mlvl_scores = np.concatenate((mlvl_scores, scores), axis=1)
        layer +=1

    scores = batch_mlvl_scores[0]
    bboxes = batch_mlvl_bboxes[0]
    box_probs = np.empty((0,6),dtype=float)
    for class_index in range(0, scores.shape[-1]):
        # person only: 1 class, multi-class: 1: 'bicycle', 2: 'bus', 3: 'car', 4: 'cat', 5: 'dog', 6: 'motorbike', 7: 'person'
        probs = scores[:,class_index]
        mask = probs >  prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = bboxes[mask, :]

        box_probs_class = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)

        box_probs_class = hard_nms(box_probs_class) #, iou_threshold, top_k, candidate_size)
        label = np.ones((box_probs_class.shape[0],1))* (class_index+1.) # adding label to box_prob
        box_probs_class = np.concatenate([box_probs_class, label], axis=1)
        box_probs = np.concatenate([box_probs, box_probs_class], axis=0)

    return box_probs

def get_bboxes(cls_scores,
               bbox_preds,
               input_shape,
               #img_metas,
               anchor_generator,
               cfg=None,
               rescale=False,
               with_nms=True,
               ):
    """Transform network output for a batch into bbox predictions.

    Args:
        cls_scores (list[Tensor]): Box scores for each level in the
            feature pyramid, has shape
            (N, num_anchors * num_classes, H, W).
        bbox_preds (list[Tensor]): Box energies / deltas for each
            level in the feature pyramid, has shape
            (N, num_anchors * 4, H, W).
        img_metas (list[dict]): Meta information of each image, e.g.,
            image size, scaling factor, etc.
        cfg (mmcv.Config | None): Test / postprocessing configuration,
            if None, test_cfg would be used
        rescale (bool): If True, return boxes in original image space.
            Default: False.
        with_nms (bool): If True, do nms before return boxes.
            Default: True.

    Returns:
        list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
            The first item is an (n, 5) tensor, where 5 represent
            (tl_x, tl_y, br_x, br_y, score) and the score between 0 and 1.
            The shape of the second tensor in the tuple is (n,), and
            each element represents the class label of the corresponding
            box.

    Example:
        >>> import mmcv
        >>> self = AnchorHead(
        >>>     num_classes=9,
        >>>     in_channels=1,
        >>>     anchor_generator=dict(
        >>>         type='AnchorGenerator',
        >>>         scales=[8],
        >>>         ratios=[0.5, 1.0, 2.0],
        >>>         strides=[4,]))
        >>> img_metas = [{'img_shape': (32, 32, 3), 'scale_factor': 1}]
        >>> cfg = mmcv.Config(dict(
        >>>     score_thr=0.00,
        >>>     nms=dict(type='nms', iou_thr=1.0),
        >>>     max_per_img=10))
        >>> feat = torch.rand(1, 1, 3, 3)
        >>> cls_score, bbox_pred = self.forward_single(feat)
        >>> # note the input lists are over different levels, not images
        >>> cls_scores, bbox_preds = [cls_score], [bbox_pred]
        >>> result_list = self.get_bboxes(cls_scores, bbox_preds,
        >>>                               img_metas, cfg)
        >>> det_bboxes, det_labels = result_list[0]
        >>> assert len(result_list) == 1
        >>> assert det_bboxes.shape[1] == 5
        >>> assert len(det_bboxes) == len(det_labels) == cfg.max_per_img
    """
    assert len(cls_scores) == len(bbox_preds)
    num_levels = len(cls_scores)
    #print("len(cls_scores): ", len(cls_scores)) 1

    featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
    #print("cls_scores[0].shape[-2:]: ", cls_scores[0].shape[-2:]) #cls_scores[0].shape[-2:]:  (12, 20)
    #mlvl_anchors = yolof_anchor_generator.grid_anchors(
    #    featmap_sizes, device='cpu')

    #mlvl_anchors = mlvl_anchors[0].detach().numpy()
    mlvl_anchors = anchor_generator.grid_anchors(
        featmap_sizes)

    #print("mlvl_anchors[0].shape: ", mlvl_anchors[0].shape) #mlvl_anchors[0].shape:  torch.Size([1200, 4])
    # Anchors in multiple feature levels. \ The sizes of each tensor should be [N, 4], where \
    # N = width * height * num_base_anchors, width and height \
    # are the sizes of the corresponding feature level, \
    # num_base_anchors is the number of anchors for that level
    # N = width * height * num_base_anchors = 20*12*5

    #mlvl_cls_scores = [cls_scores[i].detach() for i in range(num_levels)]
    #mlvl_bbox_preds = [bbox_preds[i].detach() for i in range(num_levels)]
    mlvl_cls_scores = [cls_scores]
    mlvl_bbox_preds = [bbox_preds]
    #print("mlvl_cls_scores[0].shape: ", mlvl_cls_scores[0].shape) #(1, 400, 12, 20)
    #print("mlvl_bbox_preds[0].shape: ", mlvl_bbox_preds[0].shape) #(1, 20, 12, 20)

    #img_metas:  [{'img_shape': (384, 640, 3), 'ori_shape': (384, 640, 3),
    ##'pad_shape': (384, 640, 3), 'filename': '<demo>.png', 'scale_factor': array([1., 1., 1., 1.]),
    #'flip': False, 'img_shape_for_onnx': tensor([384, 640]), 'pad_shape_for_onnx': tensor([384, 640])}]

    result_list = _get_bboxes(mlvl_cls_scores, mlvl_bbox_preds, mlvl_anchors, input_shape,)
                                  # scale_factors, cfg, rescale)
    return result_list

def intv(*value):

    if len(value) == 1:
        # one param
        value = value[0]

    if isinstance(value, tuple):
        return tuple([int(item) for item in value])
    elif isinstance(value, list):
        return [int(item) for item in value]
    elif value is None:
        return 0
    else:
        return int(value)

def drawbbox(image, box_prob, color=(153,255,55), thickness=2, textcolor=(0, 0, 0)):
    text = f"{COCO_CLASSES[int(box_prob[5])]}: {box_prob[4]:.2f}"
    x1, y1, x2, y2 = box_prob[:4]
    x = int(x1)
    y = int(y1)
    w = int(x2-x1)
    h = int(y2-y1)
    cv2.rectangle(image, (x, y, w, h), color, thickness, 16)

    border = thickness / 2
    pos = (x + 3, y - 5)
    cv2.rectangle(image, intv(x - border, y - 21, w + thickness, 21), color, -1, 16)
    #cv2.rectangle(image, intv(x - border, y - 60, w + thickness, 60), color, -1, 16)
    cv2.putText(image, text, pos, 0, 0.5, textcolor, 1, 16)

def main(args):

    # ================= parsing config start =================
    cfg_dict = mmcv.Config.fromfile(args.config_file)

    with open(args.model_cfg, 'w') as fp:
        json.dump(cfg_dict.model, fp)
    with open(args.model_cfg, 'r') as fp:
        cfg_dict_json = json.load(fp)
    anchor_cfg_from_file = cfg_dict_json['bbox_head']['anchor_generator']

    yolof_anchor_generator_infile =AnchorGenerator(ratios = anchor_cfg_from_file['ratios'],
                    scales = anchor_cfg_from_file['scales'],
                    strides = anchor_cfg_from_file['strides'],)
    # ================= parsing config end =================

    sess = onnxruntime.InferenceSession(args.onnx_file)
    providers = ['CPUExecutionProvider']
    is_cuda_available = onnxruntime.get_device() == 'GPU'
    print("is_cuda_available: ", is_cuda_available)

    if is_cuda_available:
        providers = ['CUDAExecutionProvider']
    sess.set_providers(providers)
    input_name = [item.name for item in sess.get_inputs()]

    img = args.input_img

    # ================= pre process start =================
    image = cv2.imread(img)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = cv2.resize(image , (640, 384), interpolation=cv2.INTER_LINEAR)
    image_show = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    image = image.astype(np.float32) - 128 # subtract 128 to change input data range from 0~255 to -128~127
    image = image.transpose(2, 0, 1) # CHW
    img_data = [image[None]]
    #img_data = torch.from_numpy(image)[None] #NCHW
    #print(img_data.shape)
    # ================= pre process end =================

    inf_results = sess.run(None, dict(zip(input_name, img_data)))

    # ================= post process start =================
    num_classes = 80
    normalized_cls_score, bbox_reg = inf_results

    input_shape = (args.shape[0], args.shape[1], 3)
    results = get_bboxes( cls_scores = normalized_cls_score, bbox_preds= bbox_reg, input_shape = input_shape, anchor_generator = yolof_anchor_generator_infile)
    print("results: ", results)

    for box_prob in results:
        # xywh
        drawbbox(image_show, box_prob)
    print("save result to " + args.output_img)
    cv2.imwrite(args.output_img, image_show)


if __name__ == '__main__':
    args = parse_args()
    main(args)
