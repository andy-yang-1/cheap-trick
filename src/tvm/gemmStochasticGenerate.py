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
    A_shared_block = sch.cache_read(block_Y,0,"shared")
    B_shared_block = sch.cache_read(block_Y,1,"shared")
    
    # padding for prime numbers work
    # sch.pad_einsum(block=block_Y,padding=[15,15,15])
    
    A_load_block = sch.cache_read(A_shared_block,0,"local")
    B_load_block = sch.cache_read(B_shared_block,0,"local")
    A_local_block = sch.cache_read(block_Y,0,"local")
    B_local_block = sch.cache_read(block_Y,1,"local")
    
    i, j, k = sch.get_loops(block=block_Y)

    # vbx,vtx,vtm = sch.sample_perfect_tile(loop=i,n=3,decision=[8,16,8]) 
    # vby,vty,vtn = sch.sample_perfect_tile(loop=j,n=3,decision=[8,16,8])
    # vbk,vtk = sch.sample_perfect_tile(loop=k,n=2,decision=[128,8])
    
    vbx,vtx,vtm = sch.sample_perfect_tile(loop=i,n=3) 
    vby,vty,vtn = sch.sample_perfect_tile(loop=j,n=3)
    vbk,vtk = sch.sample_perfect_tile(loop=k,n=2)

    ifactors = [vbx,vtx,vtm]
    jfactors = [vby,vty,vtn]
    kfactors = [vbk,vtk]


    bx,tx,tm = sch.split(loop=i,factors=ifactors)
    by,ty,tn = sch.split(loop=j,factors=jfactors)
    bk,tk = sch.split(loop=k,factors=kfactors)
    
    sch.reorder(by,bx,ty,tx,bk,tk,tm,tn)
    sch.bind(by,"blockIdx.y")
    sch.bind(bx,"blockIdx.x")
    sch.bind(ty,"threadIdx.y")
    sch.bind(tx,"threadIdx.x")
    
    
    sch.compute_at(A_local_block,tk,preserve_unit_loops=True)
    sch.compute_at(B_local_block,tk,preserve_unit_loops=True)
    sch.reverse_compute_at(block=C_local_block,loop=tx,preserve_unit_loops=True)
    sch.compute_at(A_shared_block,bk,preserve_unit_loops=True)
    sch.compute_at(B_shared_block,bk,preserve_unit_loops=True)
    sch.compute_at(A_load_block,bk,preserve_unit_loops=True)
    sch.compute_at(B_load_block,bk,preserve_unit_loops=True)
    
    # print(sch.mod.script())
    
    # A_shared_block = sch.cache_read(block_Y,0,"shared")
    
    by, bx, ty, tx, bk, ax0, ax1 = sch.get_loops(A_shared_block)
    ax0 = sch.fuse(ax0,ax1)
    ax0, ax1, ax2 = sch.split(loop=ax0, factors=[vty,vtx,None])
    sch.bind(ax0,"threadIdx.y")
    sch.bind(ax1,"threadIdx.x")
    
    sch.transform_layout(A_shared_block,buffer=("write",0),index_map = lambda i , j: (j,i))
    
    # B_shared_block = sch.cache_read(block_Y,1,"shared")
    
    # TODO decompose padding
    by, bx, ty, tx, bk, ax0, ax1 = sch.get_loops(B_shared_block)
    ax0 = sch.fuse(ax0,ax1)
    ax0, ax1, ax2 = sch.split(loop=ax0, factors=[vty,vtx,None])
    sch.bind(ax0,"threadIdx.y")
    sch.bind(ax1,"threadIdx.x")
    # ax0 , ax1 = sch.split(loop=ax0,factors=[tvm.tir.Mul(vtx,vty),None])
    # sch.bind(ax0,"threadIdx.x")
    
    by, bx, ty, tx, bk, tk, tm, tn = sch.get_loops(block=block_Y)
    
    
    # sch.annotate(block_or_loop=A_shared_block,ann_key="tir.manifest_shared_memory_local_stage",ann_val=1)
    # sch.annotate(block_or_loop=B_shared_block,ann_key="tir.manifest_shared_memory_local_stage",ann_val=1)
    # A_load_block = sch.cache_read(A_shared_block,0,"local")
    
    by, bx, ty, tx, bk, ax0, ax1 = sch.get_loops(A_load_block)
    ax0 = sch.fuse(ax0,ax1)
    ax0, ax1, ax2 = sch.split(loop=ax0,factors=[vty,vtx,None])
    sch.bind(ax0,"threadIdx.y")
    sch.bind(ax1,"threadIdx.x")


    # B_load_block = sch.cache_read(B_shared_block,0,"local")
    
    by, bx, ty, tx, bk, ax0, ax1 = sch.get_loops(B_load_block)
    ax0 = sch.fuse(ax0,ax1)
    ax0, ax1, ax2 = sch.split(loop=ax0,factors=[vty,vtx,None])
    sch.bind(ax0,"threadIdx.y")
    sch.bind(ax1,"threadIdx.x")
    
    # A_local_block = sch.cache_read(block_Y,0,"local")
    
    # B_local_block = sch.cache_read(block_Y,1,"local")
    
    
    sch.decompose_reduction(block=block_Y,loop=bk)
    
    # TODO add padding for some cases
    # ax0 = sch.get_loops(block=A_local_block)[-2]
    # ax1 = sch.get_loops(block=A_local_block)[-1]
    # ax0 = sch.fuse(ax0,ax1)
    # i , j = sch.split(loop=ax0,factors=[None,4])
    # ax0 = sch.get_loops(block=B_local_block)[-2]
    # ax1 = sch.get_loops(block=B_local_block)[-1]
    # ax0 = sch.fuse(ax0,ax1)
    # i , j = sch.split(loop=ax0,factors=[None,4])
    
    # print(sch.mod.script())
    
    by, bx, ty, tx, bk, tk, ax0, ax1= sch.get_loops(block=A_local_block)
    sch.unroll(ax0)
    # sch.unroll(ax1)
    by, bx, ty, tx, bk, tk, ax0, ax1= sch.get_loops(block=B_local_block)
    sch.unroll(ax0)
    # sch.unroll(ax1)
    
    # TODO add padding for some cases
    # sch.transform_layout(block=B_local_block,buffer=("read",0),index_map= lambda i,j:( i , ((j//128)*128+(j%128)//8*4+(j%8)//4*64+j%4)))
    # sch.transform_layout(block=A_local_block,buffer=("read",0),index_map= lambda j,i:( j , ((i//128)*128+(i%128)//8*4+(i%8)//4*64+i%4)))

    # sch.annotate(block_or_loop=A_shared_block,ann_key="meta_schedule.vectorize", ann_val=4)
    ax0 = sch.get_loops(block=A_shared_block)[-1]
    sch.vectorize(ax0)
    ax0 = sch.get_loops(block=B_shared_block)[-1]
    sch.vectorize(ax0)
    ax0 = sch.get_loops(block=A_local_block)[-1]
    sch.vectorize(ax0)
    ax0 = sch.get_loops(block=B_local_block)[-1]
    sch.vectorize(ax0)
    ax0 = sch.get_loops(block=A_load_block)[-1]
    sch.vectorize(ax0)
    ax0 = sch.get_loops(block=B_load_block)[-1]
    sch.vectorize(ax0)
    ax0 = sch.get_loops(block=sch.get_block(name="Y_local"))[-1]
    sch.vectorize(ax0)
    
    sch.unroll(tk)
    
    sch.annotate(block_or_loop=A_shared_block,ann_key="double_buffer_scope",ann_val=0)
    sch.annotate(block_or_loop=B_shared_block,ann_key="double_buffer_scope",ann_val=0)
    by, bx, ty, tx, bk, tk, tm, tn = sch.get_loops(block=block_Y)
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



# TODO notice padding's insertation and change index
# inst0 = sch.trace.insts[9]
# inst1 = sch.trace.insts[10]
# inst2 = sch.trace.insts[11]

# bx , tx , tm = sch.trace.decisions.get(inst0)
# by , ty , tn = sch.trace.decisions.get(inst1)
# bk , tk = sch.trace.decisions.get(inst2)
# print("Hyperparas:")
# print(by,bx,ty,tx,bk,tk,tm,tn)



    