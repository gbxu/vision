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
    # Video classification
    "r3d_18": (torchvision.models.video.r3d_18(), (1, 3, 16,112, 112), (400,)),
    "mc3_18": (torchvision.models.video.mc3_18(), (1, 3, 16,112, 112), (400,)),
    "r2plus1d_18": (torchvision.models.video.r2plus1d_18(), (1, 3, 16,112, 112), (400,))
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

class WrapperModel(torch.nn.Module):
    def __init__(self, model):
        super(WrapperModel, self).__init__()
        self._model = model
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, images, labels):
        out = self._model(images)
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
        torchvision_model, (batch_size, channels, frames, height, width), (num_classes,) = get_model[args.model_name]
        if args.batch_size > 0:
            batch_size = args.batch_size

        dummy_input = torch.randn(batch_size, channels, frames, height, width)
        dummy_labels = torch.randint(0, num_classes, (batch_size,))

        inputs = {}
        inputs['images'] = dummy_input
        inputs['labels'] = dummy_labels

        input_args = (inputs['images'],
                    inputs['labels'])
        ordered_input_names = ['images', 'labels']
        output_names = ['loss', ]
        model=WrapperModel(torchvision_model)
        # model.train()
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
        model = onnx.shape_inference.infer_shapes(model)
        onnx.checker.check_model(model)
        onnx.save(model, args.model_name+'.onnx')
