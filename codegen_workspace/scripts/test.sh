# HOME=${HOME}/gbxu/
set -v
export LD_LIBRARY_PATH=${HOME}/nccl/build/lib:$LD_LIBRARY_PATH
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)

declare -A dic
dic=(["alexnet"]=4096 ["densenet121"]=128 ["densenet169"]=128 ["densenet161"]=128 ["densenet201"]=128 ["resnet18"]=2048 ["resnet34"]=1024 ["resnet50"]=512 ["resnet101"]=256 ["resnet152"]=256 ["squeezenet1_0"]=512 ["squeezenet1_1"]=1024 ["vgg11"]=512 ["vgg13"]=256 ["vgg16"]=256 ["vgg19"]=256 ["wide_resnet50_2"]=512 ["wide_resnet101_2"]=256 )

for model_name in "alexnet" "densenet121" "densenet169" "densenet161" "densenet201" "resnet18" "resnet34" "resnet50" "resnet101" "resnet152" "squeezenet1_0" "squeezenet1_1" "vgg11" "vgg13" "vgg16" "vgg19" "wide_resnet50_2" "wide_resnet101_2"
do
    f="./"${model_name}".onnx"
    echo $f", batch:"${dic[$model_name]}
    cd ${SHELL_FOLDER}/../testmodels/${model_name}_bs${dic[$model_name]}/single_cudalib/cuda_codegen/ && rm -rf build && mkdir build && cd build && cmake .. && make -j && cp -r ../Constant ./
done

LOG_DIR="${HOME}/pita_log/"

PROFILER_CMD="nsys profile -f true -c cudaProfilerApi --stop-on-range-end=true "
# "/usr/local/cuda-10.0/bin/nvprof --profile-child-processes --profile-from-start off -o ${LOG_PATH}/%h.%p.nvvp "

echo "performance-------------------------------------------------------------------------------"

for i in `seq 3`
do
    for model_name in "alexnet" "densenet121" "densenet169" "densenet161" "densenet201" "resnet18" "resnet34" "resnet50" "resnet101" "resnet152" "squeezenet1_0" "squeezenet1_1" "vgg11" "vgg13" "vgg16" "vgg19" "wide_resnet50_2" "wide_resnet101_2"
    do
        cd ${HOME}/vision/codegen_workspace
        LOG_PATH=${LOG_DIR}"/performance/${model_name}_bs${dic[$model_name]}/single_pytorch/t${i}"
        rm -rf ${LOG_PATH} && mkdir -p ${LOG_PATH}
        echo "${LOG_PATH} ======================================"
        python3 pytorch_runtime.py --model_name $model_name --batch_size ${dic[$model_name]} > ${LOG_PATH}/result.txt

        cd ${HOME}/vision/codegen_workspace/testmodels/${model_name}_bs${dic[$model_name]}/single_cudalib/cuda_codegen/build/
        LOG_PATH=${LOG_DIR}"/performance/${model_name}_bs${dic[$model_name]}/single_cudalib/t${i}"
        rm -rf ${LOG_PATH} && mkdir -p ${LOG_PATH}
        echo "${LOG_PATH} ======================================"
        ./main_test > ${LOG_PATH}/result.txt
    done
done

echo "kernels analysis -------------------------------------------------------------------------------"

for i in `seq 3`
do
    for model_name in "alexnet" "densenet121" "densenet169" "densenet161" "densenet201" "resnet18" "resnet34" "resnet50" "resnet101" "resnet152" "squeezenet1_0" "squeezenet1_1" "vgg11" "vgg13" "vgg16" "vgg19" "wide_resnet50_2" "wide_resnet101_2"
    do
        cd ${HOME}/vision/codegen_workspace
        LOG_PATH=${LOG_DIR}"/kernels/${model_name}_bs${dic[$model_name]}/single_pytorch/t${i}"
        rm -rf ${LOG_PATH} && mkdir -p ${LOG_PATH}
        echo "${LOG_PATH} ======================================"
        ${PROFILER_CMD} -o ${LOG_PATH}/%h.%p.nvvp \
        python3 pytorch_runtime.py --model_name $model_name --batch_size ${dic[$model_name]} > ${LOG_PATH}/result.txt

        cd ${HOME}/vision/codegen_workspace/testmodels/${model_name}_bs${dic[$model_name]}/single_cudalib/cuda_codegen/build/
        LOG_PATH=${LOG_DIR}"/kernels/${model_name}_bs${dic[$model_name]}/single_cudalib/t${i}"
        rm -rf ${LOG_PATH} && mkdir -p ${LOG_PATH}
        echo "${LOG_PATH} ======================================"
        ${PROFILER_CMD} -o ${LOG_PATH}/%h.%p.nvvp \
        ./main_test > ${LOG_PATH}/result.txt
    done
done
