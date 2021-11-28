import torch
import sys
# import os
# sys.path.append(os.environ["HOME"]+"/vision/")
import torchvision # https://pytorch.org/vision/stable/models.html
from pathlib import Path
from torch.onnx import TrainingMode
import onnx
import argparse
import warnings

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default=None, help="torchvision model name")
parser.add_argument("--batch_size", type=int, default=0, help="batch size")
args = parser.parse_args()

get_model={
    # Object Detection, Instance Segmentation and Person Keypoint Detection
    "fasterrcnn_resnet50_fpn": (torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False, ), (2, 3, 600, 1000), (91, 11)), # pytorch->onnx fails: randperm
    "fasterrcnn_mobilenet_v3_large_fpn": (torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=False, pretrained_backbone=False, ), (2, 3, 600, 1000), (91, 11)), # pytorch->onnx fails: randperm
    "fasterrcnn_mobilenet_v3_large_320_fpn": (torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=False, pretrained_backbone=False, ), (2, 3, 600, 1000), (91, 11)), # pytorch->onnx fails: randperm

    "retinanet_resnet50_fpn": (torchvision.models.detection.retinanet_resnet50_fpn(pretrained=False, pretrained_backbone=False, ), (2, 3, 224, 224), (91, 11)), # pytorch->onnx fails: l1_loss 

    "ssd300_vgg16": (torchvision.models.detection.ssd300_vgg16(pretrained=False, pretrained_backbone=False, ), (4, 3, 300, 300), (91, 11)), # pytorch->onnx fails when training: smooth_l1_loss; fails when eval: resolve_conj
    "ssdlite320_mobilenet_v3_large": (torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=False, pretrained_backbone=False, ), (24, 3, 320, 320), (91, 11)), # pytorch->onnx fails: randperm; fails when eval: resolve_conj

    "maskrcnn_resnet50_fpn": (torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False, ), (2, 3, 224, 224), (91, 11)), # pytorch->onnx fails: randperm; fails when eval: resolve_conj
    # in .local/lib/python3.7/site-packages/torchvision/models/detection/transform.py: 
    #    resized_data[:, :, 0] --> resized_data[:, 0] due to no batch dimension
    # "keypointrcnn_resnet50_fpn": (torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False, trainable_backbone_layers=5), (2, 3, 224, 224), (2, 11)), # pytorch->onnx fails: randperm; fails when eval: resolve_conj
}

def infer_shapes(model, inputs, batch):
    def build_shape_dict(name, tensor, is_input, batch):
        print(name, "'s shape", tensor[0].shape)
        if isinstance(tensor, (tuple, list)):
            return [build_shape_dict(name, t, is_input, batch) for t in tensor]
        else:
            # Let's assume batch is the first axis with only 1 element (~~ might not be always true ...)
            if len(tensor.shape) > 0:
                axes = {[axis for axis, numel in enumerate(tensor.shape) if numel == batch][0]: "batch"}
            else:
                axes = {}

        print(f"Found {'input' if is_input else 'output'} {name} with shape: {axes}")
        return axes

    # Generate input names & axes
    input_dynamic_axes = {k: build_shape_dict(k, v, True, batch) for k, v in inputs.items()}
    print("input_dynamic_axes", input_dynamic_axes)

    # Generate output names & axes
    loss = model(**inputs)
    outputs = {'loss': loss}
    output_dynamic_axes = {k: build_shape_dict(k, v, False, batch) for k, v in outputs.items()}
    print("output_dynamic_axes", output_dynamic_axes)

    # Create the aggregated axes representation
    dynamic_axes = dict(input_dynamic_axes, **output_dynamic_axes)
    print("dynamic_axes:", dynamic_axes)
    return dynamic_axes

class WrapperModel(torch.nn.Module):
    def __init__(self, model):
        super(WrapperModel, self).__init__()
        self._model = model
        self.loss = torch.nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, images, targets):
        out = self._model(images, targets)
        if self.training:
            total_loss = 0
            for loss in out.values():
                total_loss += loss
            return total_loss
        else:
            total_loss = 0
            if isinstance(out, dict):
                for loss in out.values():
                    total_loss += loss
            else:
                for output in out:
                    total_loss += output.sum()
            return total_loss

if __name__ == '__main__':
    if args.model_name == None:
        model_names = get_model.keys()
    else:
        model_names = args.model_name.split(',')
    for args.model_name in model_names:
        torchvision_model, (batch_size, channels, height, width), (num_classes, ground_truth_box) = get_model[args.model_name]
        if args.batch_size > 0:
            batch_size = args.batch_size

        dummy_images = torch.randn(batch_size, channels, height, width)
        dummy_boxes = torch.zeros((batch_size, ground_truth_box, 4))
        if height < width:
            dummy_boxes[:,:,2:] = height
        else:
            dummy_boxes[:,:,2:] = width
        dummy_labels = torch.randint(1, num_classes, (batch_size, ground_truth_box))
        if args.model_name in ["maskrcnn_resnet50_fpn"]:
            dummy_masks = torch.randint(0, 1, (batch_size, 1, height, width))
        if args.model_name in ["keypointrcnn_resnet50_fpn"]:
            num_keypoints=17
            dummy_keypoints = torch.randn(batch_size, num_keypoints, 3, dtype = torch.float32) # 3: (x, y, visibility)
            dummy_keypoints[:,:,-1:] = 1
        dummy_images = list(image for image in dummy_images)
        dummy_targets = []
        for i in range(len(dummy_images)):
            d = {}
            d['boxes'] = dummy_boxes[i]
            d['labels'] = dummy_labels[i]
            if args.model_name in ["maskrcnn_resnet50_fpn"]:
                d['masks'] = dummy_masks[i]
            if args.model_name in ["keypointrcnn_resnet50_fpn"]:
                d['keypoints'] = dummy_keypoints[i]
            dummy_targets.append(d)

        inputs = {}
        inputs['images'] = dummy_images
        inputs['targets'] = dummy_targets

        input_args = (inputs['images'],
                    inputs['targets'])
        ordered_input_names = ['images', 'targets']
        output_names = ['loss', ]
        model=WrapperModel(torchvision_model)
        model.train()
        # model.eval()
        # dynamic_axes=infer_shapes(model, inputs, batch_size)
        torch.onnx.export(
            model=model,
            args=input_args,
            f=Path(args.model_name+'.onnx').as_posix(),
            input_names=ordered_input_names,
            output_names=output_names,
            # dynamic_axes=dynamic_axes,
            do_constant_folding=False,
            _retain_param_name=True,
            enable_onnx_checker=True,
            opset_version=12,
            training=TrainingMode.PRESERVE
        )

        model = onnx.load(args.model_name+'.onnx')
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model, args.model_name+'.onnx')
