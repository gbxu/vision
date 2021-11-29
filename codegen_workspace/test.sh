declare -A dic
dic=(["alexnet"]=4096 ["densenet121"]=128 ["densenet169"]=128 ["densenet161"]=128 ["densenet201"]=128 ["resnet18"]=2048 ["resnet34"]=1024 ["resnet50"]=512 ["resnet101"]=256 ["resnet152"]=256 ["squeezenet1_0"]=512 ["squeezenet1_1"]=1024 ["vgg11"]=512 ["vgg13"]=256 ["vgg16"]=256 ["vgg19"]=256 ["wide_resnet50_2"]=512 ["wide_resnet101_2"]=256 )

rm -rf testmodels
mkdir testmodels && cd testmodels

for model_name in "alexnet" "densenet121" "densenet169" "densenet161" "densenet201" "resnet18" "resnet34" "resnet50" "resnet101" "resnet152" "squeezenet1_0" "squeezenet1_1" "vgg11" "vgg13" "vgg16" "vgg19" "wide_resnet50_2" "wide_resnet101_2"
do
    python3 ../classification2onnx.py --model_name $model_name --batch_size ${dic[$model_name]}
    f="./"${model_name}".onnx"
    echo $f", batch:"${dic[$model_name]}
    # CURR_DIR=${f#*/}
    rm -rf ${model_name}_bs${dic[$model_name]}
    ${HOME}/nnfusion/build/src/tools/nnfusion/nnfusion $f \
    -f onnx -p "batch:${dic[$model_name]}" \
    -ftraining_mode=true -fautodiff=true -ftraining_optimizer="{\"optimizer\":\"SGD\",\"learning_rate\":0.01}" \
    -fblockfusion_level=0 -fenable_all_bert_fusion=true -fkernel_fusion_level=2 \
    -fextern_result_memory=true \
    -min_log_level=0 > ${model_name}_bs${dic[$model_name]}_log 2>&1
    mv nnfusion_rt ${model_name}_bs${dic[$model_name]}
done


    # -fadd_sc_allreduce=true \
