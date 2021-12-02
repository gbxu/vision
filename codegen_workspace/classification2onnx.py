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
    # # Classification
    "alexnet": (torchvision.models.alexnet(), (4096, 3, 224, 224), (1000,)),

    "densenet121": (torchvision.models.densenet121(), (128, 3, 224, 224), (1000,)),
    "densenet161": (torchvision.models.densenet161(), (128, 3, 224, 224), (1000,)),
    "densenet169": (torchvision.models.densenet169(), (128, 3, 224, 224), (1000,)),
    "densenet201": (torchvision.models.densenet201(), (128, 3, 224, 224), (1000,)),

    "efficientnet_b0": (torchvision.models.efficientnet_b0(norm_layer=torch.nn.Identity), (1, 3, 256, 224), (1000,)),
    "efficientnet_b1": (torchvision.models.efficientnet_b1(norm_layer=torch.nn.Identity), (1, 3, 256, 240), (1000,)),
    "efficientnet_b2": (torchvision.models.efficientnet_b2(norm_layer=torch.nn.Identity), (1, 3, 288, 288), (1000,)),
    "efficientnet_b3": (torchvision.models.efficientnet_b3(norm_layer=torch.nn.Identity), (1, 3, 320, 300), (1000,)),
    "efficientnet_b4": (torchvision.models.efficientnet_b4(norm_layer=torch.nn.Identity), (1, 3, 384, 380), (1000,)),
    "efficientnet_b5": (torchvision.models.efficientnet_b5(), (1, 3, 489, 456), (1000,)),
    "efficientnet_b6": (torchvision.models.efficientnet_b6(), (1, 3, 561, 528), (1000,)),
    "efficientnet_b7": (torchvision.models.efficientnet_b7(), (1, 3, 633, 600), (1000,)),

    # aux_logits=True
    "inception_v3": (torchvision.models.inception_v3(aux_logits=True), (2, 3, 299, 299), (1000,)), # batch size > 1 when training
    # "googlenet": (torchvision.models.googlenet(aux_logits=True), (1, 3, 224, 224), (1000,)), # pytorch->onnx fails: adaptive_avg_pool2d

    "mnasnet0_5": (torchvision.models.mnasnet0_5(), (1, 3, 224, 224), (1000,)),
    "mnasnet0_75": (torchvision.models.mnasnet0_75(), (1, 3, 224, 224), (1000,)),
    "mnasnet1_0": (torchvision.models.mnasnet1_0(), (1, 3, 224, 224), (1000,)),
    "mnasnet1_3": (torchvision.models.mnasnet1_3(), (1, 3, 224, 224), (1000,)),

    "mobilenet_v2": (torchvision.models.mobilenet_v2(norm_layer=torch.nn.Identity), (1, 3, 224, 224), (1000,)),
    "mobilenet_v3_large": (torchvision.models.mobilenet_v3_large(), (1, 3, 224, 224), (1000,)),
    "mobilenet_v3_small": (torchvision.models.mobilenet_v3_small(), (1, 3, 224, 224), (1000,)),

    "regnet_y_400mf": (torchvision.models.regnet_y_400mf(), (1, 3, 224, 224), (1000,)),
    "regnet_y_800mf": (torchvision.models.regnet_y_800mf(), (1, 3, 224, 224), (1000,)),
    "regnet_y_1_6gf": (torchvision.models.regnet_y_1_6gf(), (1, 3, 224, 224), (1000,)),
    "regnet_y_3_2gf": (torchvision.models.regnet_y_3_2gf(), (1, 3, 224, 224), (1000,)),
    "regnet_y_8gf": (torchvision.models.regnet_y_8gf(), (1, 3, 224, 224), (1000,)),
    "regnet_y_16gf": (torchvision.models.regnet_y_16gf(), (1, 3, 224, 224), (1000,)),
    "regnet_x_32gf": (torchvision.models.regnet_x_32gf(), (1, 3, 224, 224), (1000,)),
    "regnet_x_400mf": (torchvision.models.regnet_x_400mf(), (1, 3, 224, 224), (1000,)),
    "regnet_x_800mf": (torchvision.models.regnet_x_800mf(), (1, 3, 224, 224), (1000,)),
    "regnet_x_1_6gf": (torchvision.models.regnet_x_1_6gf(), (1, 3, 224, 224), (1000,)),
    "regnet_x_3_2gf": (torchvision.models.regnet_x_3_2gf(), (1, 3, 224, 224), (1000,)),
    "regnet_x_8gf": (torchvision.models.regnet_x_8gf(), (1, 3, 224, 224), (1000,)),
    "regnet_x_16gf": (torchvision.models.regnet_x_16gf(), (1, 3, 224, 224), (1000,)),
    "regnet_x_32gf": (torchvision.models.regnet_x_32gf(), (1, 3, 224, 224), (1000,)),

    "resnet18": (torchvision.models.resnet18(norm_layer=torch.nn.Identity), (1024, 3, 224, 224), (1000,)), # No backward for BatchNormInference in NNFusion
    "resnet34": (torchvision.models.resnet34(norm_layer=torch.nn.Identity), (1024, 3, 224, 224), (1000,)), # No backward for BatchNormInference in NNFusion
    "resnet50": (torchvision.models.resnet50(norm_layer=torch.nn.Identity), (512, 3, 224, 224), (1000,)), # No backward for BatchNormInference in NNFusion
    "resnet101": (torchvision.models.resnet101(norm_layer=torch.nn.Identity), (256, 3, 224, 224), (1000,)), # No backward for BatchNormInference in NNFusion
    "resnet152": (torchvision.models.resnet152(norm_layer=torch.nn.Identity), (256, 3, 224, 224), (1000,)), # No backward for BatchNormInference in NNFusion

    "resnext50_32x4d": (torchvision.models.resnext50_32x4d(), (1, 3, 224, 224), (1000,)),
    "resnext101_32x8d": (torchvision.models.resnext101_32x8d(), (1, 3, 224, 224), (1000,)),

    "shufflenet_v2_x0_5": (torchvision.models.shufflenet_v2_x0_5(), (1, 3, 224, 224), (1000,)),
    "shufflenet_v2_x1_0": (torchvision.models.shufflenet_v2_x1_0(), (1, 3, 224, 224), (1000,)),
    "shufflenet_v2_x1_5": (torchvision.models.shufflenet_v2_x1_5(), (1, 3, 224, 224), (1000,)),
    "shufflenet_v2_x2_0": (torchvision.models.shufflenet_v2_x2_0(), (1, 3, 224, 224), (1000,)),

    "squeezenet1_0": (torchvision.models.squeezenet1_0(), (512, 3, 224, 224), (1000,)),
    "squeezenet1_1": (torchvision.models.squeezenet1_1(), (1024, 3, 224, 224), (1000,)), # torch.nn.BatchNorm2d

    "vgg11": (torchvision.models.vgg11(), (256, 3, 224, 224), (1000,)),
    # "vgg11_bn": (torchvision.models.vgg11_bn(), (1, 3, 224, 224), (1000,)), # No backward for BatchNormInference in NNFusion
    "vgg13": (torchvision.models.vgg13(), (128, 3, 224, 224), (1000,)),
    # "vgg13_bn": (torchvision.models.vgg13_bn(), (1, 3, 224, 224), (1000,)), # No backward for BatchNormInference in NNFusion
    "vgg16": (torchvision.models.vgg16(), (128, 3, 224, 224), (1000,)),
    # "vgg16_bn": (torchvision.models.vgg16_bn(), (1, 3, 224, 224), (1000,)), # No backward for BatchNormInference in NNFusion
    "vgg19": (torchvision.models.vgg19(), (128, 3, 224, 224), (1000,)), # V100-32GB
    # "vgg19_bn": (torchvision.models.vgg19_bn(), (1, 3, 224, 224), (1000,)), # No backward for BatchNormInference in NNFusion
    
    "wide_resnet50_2": (torchvision.models.wide_resnet50_2(norm_layer=torch.nn.Identity), (256, 3, 224, 224), (1000,)),
    "wide_resnet101_2": (torchvision.models.wide_resnet101_2(norm_layer=torch.nn.Identity), (256, 3, 224, 224), (1000,)),
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
                # if is_input:
                    # if len(tensor.shape) > 2:
                        # axes[1] = "channels"
                        # axes[2] = "height"
                        # axes[3] = "width"
                    # else:
                        # axes[1] = "num_classes"
                # else:
                #     out_axes = [dim for dim, size in enumerate(tensor.shape) if size == num_classes]
                #     axes.update({dim: "num_classes" for dim in out_axes})
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

class WrapperModelAux(torch.nn.Module):
    def __init__(self, model):
        super(WrapperModelAux, self).__init__()
        self._model = model
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, images, labels):
        out = self._model(images)
        if self.training:
            # training inception_v3: logits, aux_logits1
            # training googlenet: logits, aux_logits2, aux_logits1
            total_loss = 0
            for output in out:
                total_loss += self.loss(output, labels)
            return total_loss
        else:
            loss = self.loss(out, labels)
            return loss

def get_model_with_datas(model_name, set_batch_size, infer_shape=False):
    print("get models and datas:", model_name)
    torchvision_model, (batch_size, channels, height, width), (num_classes, ) = get_model[model_name]
    if set_batch_size > 0:
        batch_size = set_batch_size
    dummy_input = torch.randn(batch_size, channels, height, width)
    dummy_labels = torch.randint(0, num_classes, (batch_size,))

    inputs = {}
    inputs['images'] = dummy_input
    inputs['labels'] = dummy_labels

    input_args = (inputs['images'],
                inputs['labels'])
    ordered_input_names = ['images', 'labels']
    output_names = ['loss', ]
    if model_name in ["inception_v3", "googlenet"]:
        model=WrapperModelAux(torchvision_model)
    else:
        model=WrapperModel(torchvision_model)
    # model.train()
    model.eval()
    if infer_shape:
        dynamic_axes=infer_shapes(model, inputs, batch_size)
    else:
        dynamic_axes=None

    return model, input_args, ordered_input_names, output_names, dynamic_axes

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
        try:
            model, input_args, ordered_input_names, output_names, dynamic_axes = get_model_with_datas(args.model_name, args.batch_size, infer_shape=True)
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
            model = onnx.shape_inference.infer_shapes(model)
            onnx.checker.check_model(model)
            onnx.save(model, args.model_name+'.onnx')
        except:
            warnings.warn("Exporting ONNX file fails:", args.model_name)
