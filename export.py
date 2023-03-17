import torch
from torch.onnx import OperatorExportTypes, register_custom_op_symbolic
from torch.onnx.symbolic_helper import parse_args

import argparse
import numpy as np
import math
import os

import nets
from utils import utils

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

parser = argparse.ArgumentParser()

# Training data
parser.add_argument('--img_height', default=768, type=int,
                    help='Image height for export')
parser.add_argument('--img_width', default=1024, type=int,
                    help='Image width for export')

# Model
parser.add_argument('--seed', default=326, type=int,
                    help='Random seed for reproducibility')
parser.add_argument('--output', type=str,
                    help='Path of output ONNX')
parser.add_argument('--opset_version', default=17,
                    type=int, help='Opset version')

# AANet
parser.add_argument('--max_disp', default=192, type=int, help='Max disparity')

parser.add_argument('--feature_type', default='aanet',
                    type=str, help='Type of feature extractor')
parser.add_argument('--no_feature_mdconv', action='store_true',
                    help='Whether to use mdconv for feature extraction')
parser.add_argument('--feature_pyramid', action='store_true',
                    help='Use pyramid feature')
parser.add_argument('--feature_pyramid_network',
                    action='store_true', help='Use FPN')
parser.add_argument('--feature_similarity', default='correlation', type=str,
                    help='Similarity measure for matching cost')
parser.add_argument('--num_downsample', default=2, type=int,
                    help='Number of downsample layer for feature extraction')
parser.add_argument('--aggregation_type', default='adaptive',
                    type=str, help='Type of cost aggregation')
parser.add_argument('--num_scales', default=3, type=int,
                    help='Number of stages when using parallel aggregation')
parser.add_argument('--num_fusions', default=6, type=int, help='Number of multi-scale fusions when using parallel'
                                                               'aggragetion')
parser.add_argument('--num_stage_blocks', default=1, type=int,
                    help='Number of deform blocks for ISA')
parser.add_argument('--num_deform_blocks', default=3, type=int,
                    help='Number of DeformBlocks for aggregation')
parser.add_argument('--no_intermediate_supervision', action='store_true',
                    help='Whether to add intermediate supervision')
parser.add_argument('--deformable_groups', default=2,
                    type=int, help='Number of deformable groups')
parser.add_argument('--mdconv_dilation', default=2, type=int,
                    help='Dilation rate for deformable conv')
parser.add_argument('--refinement_type', default='stereodrnet',
                    help='Type of refinement module')

parser.add_argument('--pretrained_aanet', default=None,
                    type=str, help='Pretrained network')

args = parser.parse_args()

model_name = os.path.basename(args.pretrained_aanet)[:-4]
model_dir = os.path.basename(os.path.dirname(args.pretrained_aanet))

if not args.output:
    args.output = os.path.join(model_dir, model_name + '.onnx')

# graph shim for torch.ops.torchvision.deform_conv2d
# hardcodes custom operator to implementation provided by mmdeploy.readthedocs.io
@parse_args('v', 'v', 'v', 'v', 'v', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i')
def deform_conv2d_symbolic(g, input, weight, offset, mask, bias, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, n_weight_grps, n_offset_grps, use_mask):
    input_tensors = [input, offset]
    assert(use_mask) # TODO otherwise use un-moduled deform conv2d
    if use_mask:
        input_tensors.append(mask)
    input_tensors.append(weight)
    if bias is not None:
        input_tensors.append(bias)
    return g.op('mmdeploy::MMCVModulatedDeformConv2d', *input_tensors,
                stride_i=(stride_h, stride_w),
                padding_i=(padding_h, padding_w),
                dilation_i=(dilation_h, dilation_w),
                deform_groups_i=n_offset_grps,
                groups_i=n_weight_grps)


register_custom_op_symbolic(
    'torchvision::deform_conv2d', deform_conv2d_symbolic, args.opset_version)


def main():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    aanet = nets.AANet(args.max_disp,
                       num_downsample=args.num_downsample,
                       feature_type=args.feature_type,
                       no_feature_mdconv=args.no_feature_mdconv,
                       feature_pyramid=args.feature_pyramid,
                       feature_pyramid_network=args.feature_pyramid_network,
                       feature_similarity=args.feature_similarity,
                       aggregation_type=args.aggregation_type,
                       num_scales=args.num_scales,
                       num_fusions=args.num_fusions,
                       num_stage_blocks=args.num_stage_blocks,
                       num_deform_blocks=args.num_deform_blocks,
                       no_intermediate_supervision=args.no_intermediate_supervision,
                       refinement_type=args.refinement_type,
                       mdconv_dilation=args.mdconv_dilation,
                       deformable_groups=args.deformable_groups).to(device)

    if os.path.exists(args.pretrained_aanet):
        print('=> Loading pretrained AANet:', args.pretrained_aanet)
        utils.load_pretrained_net(aanet, args.pretrained_aanet, no_strict=True)
    else:
        print('=> Using random initialization')

    if torch.cuda.device_count() > 1:
        print('=> Use %d GPUs' % torch.cuda.device_count())
        aanet = torch.nn.DataParallel(aanet)

    aanet.eval()

    factor = 48 if args.refinement_type != 'hourglass' else 96
    img_height = math.ceil(args.img_height / factor) * factor
    img_width = math.ceil(args.img_width / factor) * factor

    left = torch.randn(1, 3, img_height, img_width, device=device, dtype=torch.float)
    right = torch.randn(1, 3, img_height, img_width, device=device, dtype=torch.float)

    with torch.no_grad():
        torch.onnx.export(aanet.cuda(), (left, right), args.output, verbose=True, input_names=['left', 'right'],
                          output_names=['pred_disp'], operator_export_type=OperatorExportTypes.ONNX,
                          opset_version=args.opset_version)


if __name__ == '__main__':
    main()
