import argparse
import os.path as osp
import warnings

import numpy as np
import onnx
import onnxruntime as rt
import torch
from mmcv import DictAction, Config

from mmdet.core.export import (build_model_from_cfg,
                               generate_inputs_and_wrap_model,
                               preprocess_example_input)

#from onnx import optimizer
from optimizer_scripts.tools import eliminating
from optimizer_scripts.tools import fusing
from optimizer_scripts.tools import replacing
from optimizer_scripts.tools import other
from optimizer_scripts.tools import combo
from optimizer_scripts.tools import special


def torch_exported_onnx_flow(m: onnx.ModelProto, disable_fuse_bn=False) -> onnx.ModelProto:
    """Optimize the Pytorch exported onnx.

    Args:
        m (ModelProto): the input onnx model
        disable_fuse_bn (bool, optional): do not fuse BN into Conv. Defaults to False.

    Returns:
        ModelProto: the optimized onnx model
    """
    m = combo.preprocess(m, disable_fuse_bn)
    m = combo.pytorch_constant_folding(m)

    m = combo.common_optimization(m)

    m = combo.postprocess(m)

    return m

def pytorch2onnx(config_path,
                 checkpoint_path,
                 input_img,
                 input_shape,
                 opset_version=11,
                 show=False,
                 output_file='tmp.onnx',
                 verify=False,
                 normalize_cfg=None,
                 dataset='coco',
                 test_img=None,
                 do_simplify=False,
                 cfg_options=None,
                 dynamic_export=None):

    input_config = {
        'input_shape': input_shape,
        'input_path': input_img,
        'normalize_cfg': normalize_cfg
    }
    orig_output_filename = output_file
    # prepare original model and meta for verifying the onnx model
    orig_model = build_model_from_cfg(
        config_path, checkpoint_path, cfg_options=cfg_options)
    one_img, one_meta = preprocess_example_input(input_config)

    #model, tensor_data = generate_inputs_and_wrap_model(
    #    config_path, checkpoint_path, input_config,  cfg_options=cfg_options)
    model, tensor_data = generate_inputs_and_wrap_model(
        config_path, checkpoint_path, input_config, model_only = True, cfg_options=cfg_options)
    output_names = ['cls_score', 'objectness', 'bbox_reg']
    if model.with_mask:
        output_names.append('masks')
    input_name = 'input'
    dynamic_axes = None
    if dynamic_export:
        dynamic_axes = {
            input_name: {
                0: 'batch',
                2: 'width',
                3: 'height'
            },
            'dets': {
                0: 'batch',
                1: 'num_dets',
            },
            'labels': {
                0: 'batch',
                1: 'num_dets',
            },
        }
        if model.with_mask:
            dynamic_axes['masks'] = {0: 'batch', 1: 'num_dets'}

    torch.onnx.export(
        model,
        tensor_data,
        output_file,
        input_names=[input_name],
        output_names=output_names,
        export_params=True,
        keep_initializers_as_inputs=True,
        #do_constant_folding=True,
        do_constant_folding=False,
        verbose=show,
        opset_version=opset_version,
        dynamic_axes=dynamic_axes)

    m = onnx.load(output_file)
    m = torch_exported_onnx_flow(m, disable_fuse_bn = False)

    # add BN for doing input data normalization
    cfg = Config.fromfile(config_path)
    for i_n in m.graph.input:
        if i_n.type.tensor_type.shape.dim[1].dim_value != 3:
            raise ValueError("Only support 3 channel input, found input node channel not equal to 3: node name... " + i_n.name)

        mean = cfg.img_norm_cfg['mean']
        std = cfg.img_norm_cfg['std']
        normalize_bn_bias = [ -1*mean[0]/std[0], -1*mean[1]/std[1], -1*mean[2]/std[2]]
        normalize_bn_scale = [1/std[0], 1/std[1], 1/std[2]]
        other.add_shift_scale_bn_after(m.graph, i_n.name, normalize_bn_bias, normalize_bn_scale)
    m = onnx.utils.polish_model(m)

    onnx_out = output_file[:-5] + '_kneron_optimized.onnx'
    onnx.save(m, onnx_out)
    print("exported success: ", onnx_out)



def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMDetection models to ONNX')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--input-img', type=str, help='Images for input')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Show onnx graph and detection outputs')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument('--opset-version', type=int, default=11)
    parser.add_argument(
        '--test-img', type=str, default=None, help='Images for test')
    parser.add_argument(
        '--dataset', type=str, default='coco', help='Dataset name')
    parser.add_argument(
        '--verify',
        action='store_true',
        help='verify the onnx model output against pytorch output')
    parser.add_argument(
        '--simplify',
        action='store_true',
        help='Whether to simplify onnx model.')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[384, 640],
        help='input image size')
    parser.add_argument(
        '--mean',
        type=float,
        nargs='+',
        default=[128.0, 128.0, 128.0],
        help='mean value used for preprocess input data')
    parser.add_argument(
        '--std',
        type=float,
        nargs='+',
        default=[256.0, 256.0, 256.0],
        help='variance value used for preprocess input data')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--dynamic-export',
        action='store_true',
        help='Whether to export onnx with dynamic axis.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    assert args.opset_version == 11, 'MMDet only support opset 11 now'

    if not args.input_img:
        args.input_img = osp.join(
            osp.dirname(__file__), '../../tests/data/color.jpg')

    if len(args.shape) == 1:
        input_shape = (1, 3, args.shape[0], args.shape[0])
    elif len(args.shape) == 2:
        input_shape = (1, 3) + tuple(args.shape)
    else:
        raise ValueError('invalid input shape')

    assert len(args.mean) == 3
    assert len(args.std) == 3

    normalize_cfg = {'mean': args.mean, 'std': args.std}

    # convert model to onnx file
    pytorch2onnx(
        args.config,
        args.checkpoint,
        args.input_img,
        input_shape,
        opset_version=args.opset_version,
        show=args.show,
        output_file=args.output_file,
        verify=args.verify,
        normalize_cfg=normalize_cfg,
        dataset=args.dataset,
        test_img=args.test_img,
        do_simplify=args.simplify,
        cfg_options=args.cfg_options,
        dynamic_export=args.dynamic_export)
