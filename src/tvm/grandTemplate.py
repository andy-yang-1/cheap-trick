# change for general structure
import sys
import tvm
from tvm import meta_schedule as ms
from tvm.ir.module import IRModule
from tvm.script import tir as T
import numpy as np
from tvm import te
from tvm.target import Target

def stochastic_schedule(sch: tvm.tir.Schedule):

    sch.work_on("mm")
    
    block_Y = sch.get_block(name="Y")
    
    C_local_block = sch.cache_write(block_Y,0,"local")
    
    A_load_block = sch.cache_read(block_Y,0,"local")
    B_load_block = sch.cache_read(block_Y,1,"local")
    
    
    A_shared_block = sch.cache_read(block_Y,0,"shared")
    B_shared_block = sch.cache_read(block_Y,1,"shared")
    
    # padding for prime numbers work
    # sch.pad_einsum(block=block_Y,padding=[15,15,15])
    
    
    A_local_block = sch.cache_read(block_Y,0,"local")
    B_local_block = sch.cache_read(block_Y,1,"local")
    
    i, j, k = sch.get_loops(block=block_Y)    
    
    vbx,vvx,vtx,vtm1,vtm2 = sch.sample_perfect_tile(loop=i,n=5,decision=[8,1,16,1,8]) 
    vby,vvy,vty,vtn1,vtn2 = sch.sample_perfect_tile(loop=j,n=5,decision=[8,1,16,1,8])
    vbk,vtk1,vtk2 = sch.sample_perfect_tile(loop=k,n=3,decision=[128,1,8])
    
    # vbx,vvx,vtx,vtm1,vtm2 = sch.sample_perfect_tile(loop=i,n=5) 
    # vby,vvy,vty,vtn1,vtn2 = sch.sample_perfect_tile(loop=j,n=5)
    # vbk,vtk1,vtk2 = sch.sample_perfect_tile(loop=k,n=3)
    
    
    ifactors = [vbx,vvx,vtx,vtm1,vtm2]
    jfactors = [vby,vvy,vty,vtn1,vtn2]
    kfactors = [vbk,vtk1,vtk2]
    
    
    bx,vx,tx,tm1,tm2 = sch.split(loop=i,factors=ifactors)
    by,vy,ty,tn1,tn2 = sch.split(loop=j,factors=jfactors)
    bk,tk1,tk2 = sch.split(loop=k,factors=kfactors)
    
    sch.reorder(by,bx,vy,vx,ty,tx,bk,tk1,tm1,tn1,tk2,tm2,tn2)
    sch.bind(by,"blockIdx.y")
    sch.bind(bx,"blockIdx.x")
    sch.bind(vy,"vthread.y")
    sch.bind(vx,"vthread.x")
    sch.bind(ty,"threadIdx.y")
    sch.bind(tx,"threadIdx.x")
    
    sch.reverse_compute_at(C_local_block,tx,preserve_unit_loops=True)
    sch.compute_at(A_local_block,tk2,preserve_unit_loops=True)
    sch.compute_at(B_local_block,tk2,preserve_unit_loops=True)
    sch.compute_at(A_shared_block,bk,preserve_unit_loops=True)
    sch.compute_at(B_shared_block,bk,preserve_unit_loops=True)
    sch.compute_at(A_load_block,bk,preserve_unit_loops=True)
    sch.compute_at(B_load_block,bk,preserve_unit_loops=True)
    
    # sch.compute_at(A_local_block,tk2,preserve_unit_loops=True)
    # vload = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25])
    vload = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25],decision=3)
    by, bx, vy, vx, ty, tx, bk, tk1, tm1, tn1, tk2, ax0, ax1 = sch.get_loops(A_local_block)
    ax0 = sch.fuse(ax0,ax1)
    ax0, ax1 = sch.split(loop=ax0,factors=[None,vload])
    sch.unroll(ax0)
    sch.vectorize(ax1)
    
    # sch.compute_at(B_local_block,tk2,preserve_unit_loops=True)
    # vload = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25])
    vload = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25],decision=3)
    by, bx, vy, vx, ty, tx, bk, tk1, tm1, tn1, tk2, ax0, ax1 = sch.get_loops(B_local_block)
    ax0 = sch.fuse(ax0,ax1)
    ax0, ax1 = sch.split(loop=ax0,factors=[None,vload])
    sch.unroll(ax0)
    sch.vectorize(ax1)
    
    sch.transform_layout(A_shared_block,buffer=("write",0),index_map = lambda i , j: (j,i))
    
    # sch.compute_at(A_shared_block,bk,preserve_unit_loops=True)
    # vload = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25])
    vload = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25],decision=3)
    by, bx, vy, vx, ty, tx, bk, ax0, ax1 = sch.get_loops(block=A_shared_block)
    ax0 = sch.fuse(ax0,ax1)
    # ty, tx, ax0, ax1 = sch.split(loop=ax0,factors=[vty,vtx,None,vload])
    ax0, ty, tx, ax1 = sch.split(loop=ax0,factors=[None,vty,vtx,vload])
    # sch.unroll(ax0)
    sch.bind(ty,"threadIdx.y")
    sch.bind(tx,"threadIdx.x")
    sch.vectorize(ax1)
    
    # sch.compute_at(B_shared_block,bk,preserve_unit_loops=True)
    # vload = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25])
    vload = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25],decision=3)
    by, bx, vy, vx, ty, tx, bk, ax0, ax1 = sch.get_loops(block=B_shared_block)
    ax0 = sch.fuse(ax0,ax1)
    # ty, tx, ax0, ax1 = sch.split(loop=ax0,factors=[vty,vtx,None,vload])
    ax0, ty, tx, ax1 = sch.split(loop=ax0,factors=[None,vty,vtx,vload])
    # sch.unroll(ax0)
    sch.bind(ty,"threadIdx.y")
    sch.bind(tx,"threadIdx.x")
    sch.vectorize(ax1)
    
    
    # A_load_block = sch.cache_read(A_shared_block,0,storage_scope="local")
    # sch.compute_at(A_load_block,bk,preserve_unit_loops=True)
    # vload = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25])
    vload = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25],decision=3)
    by, bx, vy, vx, ty, tx, bk, ax0, ax1 = sch.get_loops(block=A_load_block)
    ax0 = sch.fuse(ax0,ax1)
    # ty, tx, ax0, ax1 = sch.split(loop=ax0,factors=[vty,vtx,None,vload])
    ax0, ty, tx, ax1 = sch.split(loop=ax0,factors=[None,vty,vtx,vload])
    # sch.unroll(ax0)
    sch.bind(ty,"threadIdx.y")
    sch.bind(tx,"threadIdx.x")
    sch.vectorize(ax1)
    
    sch.transform_layout(block=B_local_block,buffer=("read",0),index_map= lambda i,j:( i , ((j//128)*128+(j%128)//8*4+(j%8)//4*64+j%4)))
    sch.transform_layout(block=A_local_block,buffer=("read",0),index_map= lambda j,i:( j , ((i//128)*128+(i%128)//8*4+(i%8)//4*64+i%4)))

    
    # B_load_block = sch.cache_read(B_shared_block,0,storage_scope="local")
    # sch.compute_at(B_load_block,bk,preserve_unit_loops=True)
    # vload = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25])
    vload = sch.sample_categorical(candidates=[1, 2, 3, 4], probs=[0.25, 0.25, 0.25, 0.25],decision=3)
    by, bx, vy, vx, ty, tx, bk, ax0, ax1 = sch.get_loops(block=B_load_block)
    ax0 = sch.fuse(ax0,ax1)
    # ty, tx, ax0, ax1 = sch.split(loop=ax0,factors=[vty,vtx,None,vload])
    ax0, ty, tx, ax1 = sch.split(loop=ax0,factors=[None,vty,vtx,vload])
    # sch.unroll(ax0)
    sch.bind(ty,"threadIdx.y")
    sch.bind(tx,"threadIdx.x")
    sch.vectorize(ax1)
    
    by, bx, vy, vx, ty, tx, bk, tk1, tm1, tn1, tk2, tm2, tn2 = sch.get_loops(block=block_Y)
    sch.unroll(tk1)
    sch.unroll(tk2)
    # vunroll = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=3)
    # sch.annotate(block_or_loop=bk,ann_key="pragma_auto_unroll_max_step",ann_val=vunroll)
    
    sch.decompose_reduction(block=block_Y,loop=bk)
    
    sch.annotate(block_or_loop=A_shared_block,ann_key="double_buffer_scope",ann_val=0)
    sch.annotate(block_or_loop=B_shared_block,ann_key="double_buffer_scope",ann_val=0)
    by, bx, vy, vx, ty, tx, bk, tk1, tm1, tn1, tk2, tm2, tn2 = sch.get_loops(block=block_Y)
    sch.annotate(block_or_loop=bk,ann_key="software_pipeline_stage",ann_val=[0,0,0,0,1])
    sch.annotate(block_or_loop=bk,ann_key="software_pipeline_order",ann_val=[0,3,1,4,2])
    
    
    
    return sch
    
    
    
    
M = int(sys.argv[1])
N = int(sys.argv[2])
K = int(sys.argv[3]) 
t = int(sys.argv[4])   

# M = int(input())
# N = int(input())
# K = int(input())  


A = te.placeholder((M,K),"float32",name="A")
B = te.placeholder((K,N),"float32",name="B")
k = te.reduce_axis((0,K),name="k")
Y = te.compute((M,N), lambda i , j: te.sum(A[i,k]*B[k,j],axis=k),name="Y")
    
# A = te.placeholder((1024,1024),"float32",name="A")
# B = te.placeholder((1024,1024),"float32",name="B")
# k = te.reduce_axis((0,1024),name="k")
# Y = te.compute((1024,1024), lambda i , j: te.sum(A[i,k]*B[k,j],axis=k),name="Y")

sgemm_func = te.create_prim_func([A,B,Y]).with_attr({"global_symbol":"mm"})
myModule = tvm.IRModule({"mm":sgemm_func})

# sch = tvm.tir.Schedule(myModule)
# sch = stochastic_schedule(sch)


db = ms.tir_integration.tune_tir(
    mod=myModule,
    target="nvidia/geforce-rtx-3080",
    work_dir="./db/stochastic",
    space=ms.space_generator.ScheduleFn(
        stochastic_schedule,
        sch_rules=[],
        postprocs=[],
        mutator_probs={},
    ),
    # strategy=ms.search_strategy.ReplayFunc(),
    max_trials_global=t,
    # runner="local",
    runner=ms.runner.RPCRunner(  # type: ignore
                rpc_config=ms.runner.RPCConfig(
                tracker_host="172.16.2.241",
                tracker_port=4445,
                tracker_key="rtx-3080",
                session_timeout_sec=600,
                ),
                # evaluator_config=ms.runner.EvaluatorConfig(
                #     number=ARGS.number,
                #     repeat=ARGS.repeat,
                #     min_repeat_ms=ARGS.min_repeat_ms,
                #     enable_cpu_cache_flush=ARGS.cpu_flush,
                # ),
                alloc_repeat=1,
            ),
)

sch = db.query_schedule(myModule, target=Target("nvidia/geforce-rtx-3080"), workload_name="main") 

sfile = open("script.out","w")
print(sch.mod.script(),file=sfile)
sfile.flush()
sfile.close()

rt_mod = tvm.build(sch.mod,target="cuda")
# print(rt_mod.imported_modules[0].get_source())
tfile = open("gemm.cu","w")
print(rt_mod.imported_modules[0].get_source(),file=tfile)
tfile.flush()
tfile.close()

a_np = np.random.rand(M, K).astype(A.dtype)
b_np = np.random.rand(K, N).astype(B.dtype)
c_np = a_np @ b_np
a_tvm = tvm.nd.array(a_np, device=tvm.cuda(0))
b_tvm = tvm.nd.array(b_np, device=tvm.cuda(0))
c_tvm = tvm.nd.array(np.empty((M, N)).astype(Y.dtype), device=tvm.cuda(0))
rt_mod(a_tvm, b_tvm, c_tvm)
assert np.allclose(c_tvm.numpy(), c_np)

time_f = rt_mod.time_evaluator(rt_mod.entry_name, dev=tvm.cuda(0), number=100)
time = time_f(a_tvm, b_tvm, c_tvm).mean

flop = M * N * K * 2
print("GFLOPS: %.2f" % (flop / time / 1e9))