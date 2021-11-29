
for model_name in "alexnet" "densenet121" "densenet169" "densenet161" "densenet201" "resnet18" "resnet34" "resnet50" "resnet101" "resnet152" "squeezenet1_0" "squeezenet1_1" "vgg11" "vgg13" "vgg16" "vgg19" "wide_resnet50_2" "wide_resnet101_2"
do
    echo model_name=$model_name
    python3 pytorch_runtime.py --model_name $model_name > $model_name.txt 2>&1
done
