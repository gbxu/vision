# HOME=/home/ubuntu/gbxu/
MASTER="worker-0"
NODE_LIST=("worker-1")
MPI_CMD="
    /usr/local/bin/mpirun \
    --allow-run-as-root \
    -bind-to none -map-by slot \
    -mca pml ob1 -mca btl ^openib \
    -mca btl_tcp_if_include ib1 \
    -tag-output -merge-stderr-to-stdout \
    -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
    -x NCCL_DEBUG=INFO \
    -x NCCL_SOCKET_IFNAME=ib0,ib1 -x NCCL_IB_DISABLE=0 -x NCCL_IB_HCA=mlx5 -x NCCL_P2P_DISABLE=0 -x NCCL_IB_CUDA_SUPPORT=1
    "

# MPI_CMD="
#     /usr/local/bin/mpirun \
#     --allow-run-as-root \
#     -mca pml ob1 \
#     -mca btl ^openib \
#     -mca btl_tcp_if_include eth0 \
#     -tag-output -merge-stderr-to-stdout \
#     -x PATH -x LD_LIBRARY_PATH -x PYTHONPATH \
#     -x NCCL_DEBUG=INFO \
#     -x NCCL_IB_GID_INDEX=3 \
#     -x NCCL_SOCKET_IFNAME=bond0 -x NCCL_IB_HCA=mlx5
#     "

###################################################################
set -v
export LD_LIBRARY_PATH=${HOME}/nccl/build/lib:$LD_LIBRARY_PATH
SHELL_FOLDER=$(cd "$(dirname "$0")";pwd)

GPU_PER_NODE=8
NUM_NODES=$((${#NODE_LIST[@]}+1))

declare -A dic
dic=(["alexnet"]=4096 ["densenet121"]=128 ["densenet169"]=128 ["densenet161"]=128 ["densenet201"]=128 ["resnet18"]=2048 ["resnet34"]=1024 ["resnet50"]=512 ["resnet101"]=256 ["resnet152"]=256 ["squeezenet1_0"]=512 ["squeezenet1_1"]=1024 ["vgg11"]=512 ["vgg13"]=256 ["vgg16"]=256 ["vgg19"]=256 ["wide_resnet50_2"]=512 ["wide_resnet101_2"]=256 )

############ mpi options ############
ALL_GPUS=$(($GPU_PER_NODE*$NUM_NODES))
ALL_NODES=${MASTER}:$GPU_PER_NODE
for node in $NODE_LIST
do
    ALL_NODES=$ALL_NODES,$node:$GPU_PER_NODE
done

LOG_DIR="${HOME}/vision_log/"

PROFILER_CMD="nsys profile -f true -c cudaProfilerApi --stop-on-range-end=true "
# "/usr/local/cuda-10.0/bin/nvprof --profile-child-processes --profile-from-start off -o ${LOG_PATH}/%h.%p.nvvp "

echo "performance-------------------------------------------------------------------------------"

for node in $NODE_LIST
do
    scp -r ${SHELL_FOLDER}/../*.py $node:${SHELL_FOLDER}/../
done

for i in `seq 3`
do
    for model_name in "alexnet" "densenet121" "densenet169" "densenet161" "densenet201" "resnet18" "resnet34" "resnet50" "resnet101" "resnet152" "squeezenet1_0" "squeezenet1_1" "vgg11" "vgg13" "vgg16" "vgg19" "wide_resnet50_2" "wide_resnet101_2"
    do
        cd ${HOME}/vision/codegen_workspace
        LOG_PATH=${LOG_DIR}"/performance/${model_name}_bs${dic[$model_name]}/dp_pytorch/${NUM_NODES}x${GPU_PER_NODE}V100/t${i}"
        rm -rf ${LOG_PATH} && mkdir -p ${LOG_PATH}
        echo "${LOG_PATH} ======================================"
        ${MPI_CMD} -output-filename ${LOG_PATH} -H ${ALL_NODES} -np ${ALL_GPUS} \
        python3 pytorch_runtime.py --master_ip ${MASTER} --model_name $model_name --batch_size ${dic[$model_name]} > ${LOG_PATH}/result.txt
    done
done

echo "contention analysis -------------------------------------------------------------------------------"

for i in `seq 3`
do
    for model_name in "alexnet" "densenet121" "densenet169" "densenet161" "densenet201" "resnet18" "resnet34" "resnet50" "resnet101" "resnet152" "squeezenet1_0" "squeezenet1_1" "vgg11" "vgg13" "vgg16" "vgg19" "wide_resnet50_2" "wide_resnet101_2"
    do
        cd ${HOME}/vision/codegen_workspace
        LOG_PATH=${LOG_DIR}"/kernels/${model_name}_bs${dic[$model_name]}/dp_pytorch/${NUM_NODES}x${GPU_PER_NODE}V100/t${i}"
        rm -rf ${LOG_PATH} && mkdir -p ${LOG_PATH}
        echo "${LOG_PATH} ======================================"
        ${MPI_CMD} -output-filename ${LOG_PATH} -H ${ALL_NODES} -np ${ALL_GPUS} \
        ${PROFILER_CMD} -o ${LOG_PATH}/%h.%p.nvvp \
        python3 pytorch_runtime.py --master_ip ${MASTER} --model_name $model_name --batch_size ${dic[$model_name]} > ${LOG_PATH}/result.txt
    done
done

exit

############ build code; distribute exe ############
for model_name in "alexnet" "densenet121" "densenet169" "densenet161" "densenet201" "resnet18" "resnet34" "resnet50" "resnet101" "resnet152" "squeezenet1_0" "squeezenet1_1" "vgg11" "vgg13" "vgg16" "vgg19" "wide_resnet50_2" "wide_resnet101_2"
do
    f="./"${model_name}".onnx"
    echo $f", batch:"${dic[$model_name]}
    cd ${SHELL_FOLDER}/../testmodels/${model_name}_bs${dic[$model_name]}/dp_cudalib/cuda_codegen/ && rm -rf build && mkdir build && cd build && cmake .. && make -j && cp -r ../Constant ./
done

for node in $NODE_LIST
do
    scp -r ${SHELL_FOLDER}/../testmodels $node:${SHELL_FOLDER}/../
done

echo "performance-------------------------------------------------------------------------------"

for i in `seq 3`
do
    for model_name in "alexnet" "densenet121" "densenet169" "densenet161" "densenet201" "resnet18" "resnet34" "resnet50" "resnet101" "resnet152" "squeezenet1_0" "squeezenet1_1" "vgg11" "vgg13" "vgg16" "vgg19" "wide_resnet50_2" "wide_resnet101_2"
    do
        cd ${HOME}/vision/codegen_workspace/testmodels/${model_name}_bs${dic[$model_name]}/dp_cudalib/cuda_codegen/build/
        LOG_PATH=${LOG_DIR}"performance/${model_name}_bs${dic[$model_name]}/dp_cudalib/${NUM_NODES}x${GPU_PER_NODE}V100/t${i}"
        rm -rf ${LOG_PATH} && mkdir -p ${LOG_PATH}
        echo "${LOG_PATH} ======================================"
        ${MPI_CMD} -output-filename ${LOG_PATH} -H ${ALL_NODES} -np ${ALL_GPUS} \
        ./main_test > ${LOG_PATH}/result.txt
    done
done

echo "contention analysis -------------------------------------------------------------------------------"

for i in `seq 3`
do
    for model_name in "alexnet" "densenet121" "densenet169" "densenet161" "densenet201" "resnet18" "resnet34" "resnet50" "resnet101" "resnet152" "squeezenet1_0" "squeezenet1_1" "vgg11" "vgg13" "vgg16" "vgg19" "wide_resnet50_2" "wide_resnet101_2"
    do
        cd ${HOME}/vision/codegen_workspace/testmodels/${model_name}_bs${dic[$model_name]}/dp_cudalib/cuda_codegen/build/
        LOG_PATH=${LOG_DIR}"kernels/${model_name}_bs${dic[$model_name]}/dp_cudalib/${NUM_NODES}x${GPU_PER_NODE}V100/t${i}"
        rm -rf ${LOG_PATH} && mkdir -p ${LOG_PATH}
        echo "${LOG_PATH} ======================================"
        ${MPI_CMD} -output-filename ${LOG_PATH} -H ${ALL_NODES} -np ${ALL_GPUS} \
        ${PROFILER_CMD} -o ${LOG_PATH}/%h.%p.nvvp \
        ./main_test > ${LOG_PATH}/result.txt
    done
done
