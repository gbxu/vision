import torch
from torch.optim import SGD
from torchinfo import summary
import time
import argparse
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from classification2onnx import get_model_with_datas as get_model_with_datas_classification

def pytorch_train():
    if args.model_type == "classification":
        model, input_args, ordered_input_names, output_names, dynamic_axes = get_model_with_datas_classification(args.model_name, args.batch_size)
    else:
        raise Exception("other model types are not supported so far.")
    # should switch to train(), other than eval()
    model.eval()
    model.to(device)
    (images, labels) = input_args
    images = images.to(device)
    labels = labels.to(device)


    optimizer = SGD(model.parameters(), lr=1e-3)

    if args.jit:
        model = torch.jit.trace(model, input_args)

    gradient_idx = {}
    parameters = [p for p in model.parameters() if p.requires_grad]
    for idx, p in enumerate(parameters):
        gradient_idx[p] = idx
    # dist._DEFAULT_FIRST_BUCKET_BYTES = 1MB
    # https://github.com/pytorch/pytorch/blob/2ca552160b23be5d1bf2accc23733a2066a78899/torch/csrc/distributed/c10d/reducer.cpp#L1951
    bucket_size_list = []
    bucket_shape_list = []
    bucket_idx_list = []
    if args.world_size > 1:
        def record_bucket_size(state, bucket):
            bucket_size = 0
            curr_bucket_shapes = []
            curr_bucket_idx = []
            for (tensor, parameter) in zip(reversed(bucket.gradients()), reversed(bucket.parameters())):
                curr_bucket_idx.append(gradient_idx[parameter])
                curr_bucket_shapes.append(tensor.shape)
                size = 1
                for i in tensor.shape:
                    size *= i
                bucket_size += size
            bucket_size_list.append(bucket_size)
            bucket_shape_list.append(curr_bucket_shapes)
            bucket_idx_list.append(curr_bucket_idx)
            fut = torch.futures.Future()
            fut.set_result(bucket.buffer())
            return fut

        init_method = 'tcp://'
        init_method += args.master_ip + ':' + args.master_port
        torch.distributed.init_process_group(backend='nccl',
                                            rank=args.rank,
                                            world_size=args.world_size,
                                            init_method=init_method)
        model = DDP(model, device_ids=[args.local_rank], bucket_cap_mb=args.allreduce_buffer, output_device=args.local_rank)
        model.register_comm_hook(state=None, hook=record_bucket_size)

    # record ddp bucket actual num of floats
    # buckets will be rebulit in the 2nd iteration.
    # https://github.com/pytorch/pytorch/blob/f6696c5a85bdc19ecd97e427c3b847e661d3fcfc/torch/csrc/distributed/c10d/reducer.cpp#L1651
    init_bucket_size_list = []
    init_bucket_shape_list = []
    init_bucket_idx_list = []
    # warmup
    for i in range(10):
        bucket_size_list.clear()
        bucket_shape_list.clear()
        bucket_idx_list.clear()
        optimizer.zero_grad()
        loss = model(images, labels)
        loss.backward()
        optimizer.step()
        if args.rank == 0 :
            print("%s step %d: bucket_size_list:%s, bucket_shape_list:%s, bucket_idx_list:%s"%(args.model_name, i, bucket_size_list, bucket_shape_list, bucket_idx_list))
        if i == 0:
            init_bucket_size_list = bucket_size_list
            init_bucket_shape_list = bucket_shape_list
            init_bucket_idx_list = bucket_idx_list

    repeat = args.repeat
    torch.cuda.cudart().cudaProfilerStart()
    start = time.time()
    for i in range(repeat):
        bucket_size_list.clear()
        bucket_shape_list.clear()
        bucket_idx_list.clear()
        torch.cuda.nvtx.range_push('iteration')
        optimizer.zero_grad()
        loss = model(images, labels)
        loss.backward()
        optimizer.step()
        torch.cuda.nvtx.range_pop()
    end = time.time()
    if args.world_size > 1:
        torch.distributed.barrier()
    if args.rank == 0 :
        print("pytorch train %s:"%(args.model_name), (end-start)/repeat*1000, "ms/iter.")

    # pure allreduc test
    # if args.rank == 0 :
    #     print("== init iteration")
    # if args.world_size > 1:
    #     torch.distributed.barrier()
    # for idx, bucket_size in enumerate(init_bucket_size_list):
    #     tensor = torch.randn(1, bucket_size).to(args.local_rank)
    #     repeat = 50
    #     start = time.time()
    #     for i in range(repeat):
    #         torch.cuda.nvtx.range_push('init pure_allreduce')
    #         torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    #         torch.cuda.synchronize(args.local_rank)
    #         torch.cuda.nvtx.range_pop()
    #     end = time.time()
    #     torch.distributed.barrier()
    #     duration=(end-start)/repeat*1000
    #     if args.rank == 0 :
    #         print("pytorch pure allreduce idx=%d, floats=%d, bytes=%d, duration=%fms, bw=%fGB/s"%(idx, bucket_size, bucket_size*4, duration, bucket_size*4/1024/1024/1024/(duration/1000)))


    if args.world_size > 1:
        if args.rank == 0 :
            print("== runtime iteration")
        torch.distributed.barrier()
        for idx, bucket_size in enumerate(bucket_size_list):
            tensor = torch.randn(1, bucket_size).to(args.local_rank)
            repeat = args.repeat
            start = time.time()
            for i in range(repeat):
                torch.cuda.nvtx.range_push('runtime pure_allreduce')
                torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
                torch.cuda.synchronize(args.local_rank)
                torch.cuda.nvtx.range_pop()
            end = time.time()
            torch.distributed.barrier()
            duration=(end-start)/repeat*1000
            if args.rank == 0 :
                print("pytorch pure allreduce idx=%d, floats=%d, bytes=%d, duration=%fms, bw=%fGB/s"%(idx, bucket_size, bucket_size*4, duration, bucket_size*4/1024/1024/1024/(duration/1000)))
        torch.distributed.barrier()

    torch.cuda.cudart().cudaProfilerStop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=50, help="")
    parser.add_argument("--rank", type=int, default=0, help="global rank")
    parser.add_argument("--model_type", type=str, default="classification", help="classification, segmentation, video, detection")
    parser.add_argument("--model_name", type=str, default=None, help="torchvision model name")
    parser.add_argument("--batch_size", type=int, default=0, help="batch size")
    parser.add_argument("--local_rank", type=int, default=0, help="local rank")
    parser.add_argument("--world_size", type=int, default=1, help="world size")
    parser.add_argument("--master_ip", type=str, default="localhost", help="master ip")
    parser.add_argument("--master_port", type=str, default="6000", help="master port")
    parser.add_argument("-m", "--allreduce_buffer", type=int, default=25, help="MB of allreduce_buffer")
    parser.add_argument("--jit", action='store_true', default=False, help="enable torhc.jit")
    args = parser.parse_args()
    args.rank = int(os.environ.get('OMPI_COMM_WORLD_RANK', args.rank))
    args.local_rank = int(os.environ.get('OMPI_COMM_WORLD_LOCAL_RANK', args.local_rank))
    args.world_size = int(os.environ.get('OMPI_COMM_WORLD_SIZE', args.world_size))
    device = args.local_rank
    torch.cuda.set_device(device)
    if args.rank == 0:
        print(args)

    pytorch_train()
