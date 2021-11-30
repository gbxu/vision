from os import WEXITED
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

get_model={
    # Semantic Segmentation
    "fcn_resnet50": (torchvision.models.segmentation.fcn_resnet50(aux_loss=True, pretrained_backbone=False), (1, 3, 520, 520), (21,) ),
    "fcn_resnet101": (torchvision.models.segmentation.fcn_resnet101(aux_loss=True, pretrained_backbone=False), (1, 3, 520, 520), (21,) ),

    "deeplabv3_resnet50": (torchvision.models.segmentation.deeplabv3_resnet50(aux_loss=True, pretrained_backbone=False), (2, 3, 520, 520), (21,) ), # batch size > 1
    "deeplabv3_resnet101": (torchvision.models.segmentation.deeplabv3_resnet101(aux_loss=True, pretrained_backbone=False), (2, 3, 520, 520), (21,) ), # batch size > 1
    "deeplabv3_mobilenet_v3_large": (torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(aux_loss=True, pretrained_backbone=False), (2, 3, 520, 520), (21,) ), # batch size > 1

    "lraspp_mobilenet_v3_large": (torchvision.models.segmentation.lraspp_mobilenet_v3_large(pretrained_backbone=False), (1, 3, 520, 520), (21,) ), # Unsupported: HardSigmoid
}

def infer_shapes(model, inputs, batch):
    def build_shape_dict(name, tensor, is_input, batch):
        print(name, "'s shape", tensor.shape)
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

class WrapperModelAux(torch.nn.Module):
    def __init__(self, model):
        super(WrapperModelAux, self).__init__()
        self._model = model
        self.loss = torch.nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, images, labels):
        out = self._model(images)
        if isinstance(out, dict):
            loss = self.loss(out['aux'], labels)
            return loss
        else:
            loss = self.loss(out, labels)
            return loss

class WrapperModelOut(torch.nn.Module):
    def __init__(self, model):
        super(WrapperModelOut, self).__init__()
        self._model = model
        self.loss = torch.nn.BCEWithLogitsLoss(reduction='sum')

    def forward(self, images, labels):
        out = self._model(images)['out']
        loss = self.loss(out, labels)
        return loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None, help="torchvision model name")
    parser.add_argument("--batch_size", type=int, default=0, help="batch size")
    args = parser.parse_args()

    if args.model_name == None:
        model_names = get_model.keys()
    else:
        model_names = args.model_name.split(',')
    for args.model_name in model_names:
        torchvision_model, (batch_size, channels, height, width), (num_classes, ) = get_model[args.model_name]
        if args.batch_size > 0:
            batch_size = args.batch_size

        dummy_input = torch.randn(batch_size, channels, height, width)
        dummy_labels = torch.randn(batch_size, num_classes, height, width) # target object

        inputs = {}
        inputs['images'] = dummy_input
        inputs['labels'] = dummy_labels

        input_args = (inputs['images'],
                    inputs['labels'])
        ordered_input_names = ['images', 'labels']
        output_names = ['loss', ]
        if args.model_name in ["lraspp_mobilenet_v3_large"]:
            model=WrapperModelOut(torchvision_model)
        else:
            model=WrapperModelAux(torchvision_model)
        # model.train() # No Support training version Dropout in NNFusion
        model.eval()
        dynamic_axes=infer_shapes(model, inputs, batch_size)
        torch.onnx.export(
            model=model,
            args=input_args,
            f=Path(args.model_name+'.onnx').as_posix(),
            input_names=ordered_input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=False,
            _retain_param_name=True,
            enable_onnx_checker=True,
            opset_version=12,
            training=TrainingMode.PRESERVE
        )

        model = onnx.load(args.model_name+'.onnx')

        for idx, node in enumerate(model.graph.node):
            if node.op_type in ["MaxPool", "Dropout", "BatchNormalization", "BatchNormInference"]:
                # node.op_type = "AveragePool"
                # raise NotImplementedError("NNFusion does'nt support %s so far."%(node.op_type))
                warnings.warn("NNFusion does'nt support %s so far."%(node.op_type))
            # if node.op_type == "Resize":
            #     static_size_node = onnx.helper.make_node(
            #         name="static_"+node.input[3],
            #         op_type='Constant',
            #         inputs=[],
            #         outputs=["static_"+node.input[3]+'_output'],
            #         value=onnx.helper.make_tensor(
            #             name='resize_sizes',
            #             data_type=onnx.TensorProto.INT64,
            #             dims=(4,),
            #             vals=(batch_size, num_classes, height, width),
            #         ),
            #     )
            #     model.graph.node.insert(idx-1, static_size_node)
            #     new_resize_node = onnx.helper.make_node(
            #         name="new_"+node.name,
            #         op_type=node.op_type,
            #         inputs=[node.input[0], '', '', static_size_node.name+'_output'],
            #         # inputs=[node.input[0], node.input[1], node.input[2], static_size_node.name+'_output'],
            #         outputs=node.output,
            #         coordinate_transformation_mode=node.attribute[0].s,
            #         cubic_coeff_a=node.attribute[1].f,
            #         mode=node.attribute[2].s,
            #         nearest_mode=node.attribute[3].s,
            #     )
            #     model.graph.node.remove(node)
            #     model.graph.node.insert(idx, new_resize_node)
        # onnx.checker.check_model(model)
        model = onnx.shape_inference.infer_shapes(model)
        onnx.save(model, args.model_name+'.onnx')
