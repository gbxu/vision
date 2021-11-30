rm -rf data
mkdir data
cd data

mkdir classification
cd classification
python3 ../../classification2onnx.py > log 2>&1
cd ..

mkdir object_detection
cd object_detection
python3 ../../detection2onnx.py > log 2>&1
cd ..

mkdir semantic_segmentation
cd semantic_segmentation
python3 ../../segmentation2onnx.py > log 2>&1
cd ..

mkdir video_classification
cd video_classification
python3 ../../video2onnx.py > log 2>&1
cd ..

cd ..

# WORKLOADS=("./data/classification/*.onnx"  )
# WORKLOADS=("./data/semantic_segmentation/*.onnx" )
# WORKLOADS=("./data/video_classification/*.onnx")
WORKLOADS=("./data/classification/*.onnx" "./data/semantic_segmentation/*.onnx" "./data/object_detection/*.onnx" "./data/video_classification/*.onnx")
WORKLOADS=${WORKLOADS[*]}
for WORKLOAD in $WORKLOADS
do
    echo $WORKLOAD
    for f in $WORKLOAD
    do
        echo $f
        CURR_DIR="./codegen/"${f#*/}
        rm -rf $CURR_DIR
        mkdir -p $CURR_DIR
        ${HOME}/nnfusion/build/src/tools/nnfusion/nnfusion $f \
        -f onnx -p "batch:8" \
        -ftraining_mode=true -fautodiff=true -ftraining_optimizer="{\"optimizer\":\"SGD\",\"learning_rate\":0.01}" \
        -fblockfusion_level=0 -fenable_all_bert_fusion=true -fkernel_fusion_level=2 \
        -fextern_result_memory=true \
        -fadd_sc_allreduce=true \
        -min_log_level=0 > $CURR_DIR/temp 2>&1
        mv nnfusion_rt $CURR_DIR/
    done
done