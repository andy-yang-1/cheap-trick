import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T
# from tvm import relax
import numpy as np
from tvm import te

# This is needed for deferring annotation parsing in TVMScript
# from __future__ import annotations 

# initialize
A = te.placeholder((1024,1024),"float32",name="A")
B = te.placeholder((1024,1024),"float32",name="B")
k = te.reduce_axis((0,1024),name="k")
Y = te.compute((1024,1024), lambda i , j: te.sum(A[i,k]*B[k,j],axis=k),name="Y")

sgemm_func = te.create_prim_func([A,B,Y]).with_attr({"global_symbol":"mm"})
myModule = tvm.IRModule({"mm":sgemm_func})
sch = tvm.tir.Schedule(myModule)
sch.work_on("mm")

# add A_shared & B_shared & C_local
block_Y = sch.get_block(name="Y") 
C_local_block = sch.cache_write(block_Y,0,"local")
B_shared_block = sch.cache_read(block_Y,1,"shared")
A_shared_block = sch.cache_read(block_Y,0,"shared")

# add A_local & B_local
B_local_block = sch.cache_read(block_Y,1,"local")
A_local_block = sch.cache_read(block_Y,0,"local")

# add local load stage
A_load_block = sch.cache_read(A_shared_block,0,"local")
B_load_block = sch.cache_read(B_shared_block,0,"local")

# transpose A
sch.transform_layout(A_shared_block,buffer=("write",0),index_map = lambda i , j: (j,i))

# bind the threads
i, j, k = sch.get_loops(block=block_Y)
bx,tx,tm = sch.split(loop=i,factors=[8,16,8])
by,ty,tn = sch.split(loop=j,factors=[8,16,8])
bk,tk = sch.split(loop=k,factors=[128,8])
sch.bind(bx,"blockIdx.x")
sch.bind(by,"blockIdx.y")
sch.bind(tx,"threadIdx.x")
sch.bind(ty,"threadIdx.y")
sch.reorder(by,bx,ty,tx,bk,tk,tm,tn)

# decompose the initial block
by, bx, ty, tx, bk, tk, tm, tn = sch.get_loops(block=block_Y)
sch.decompose_reduction(block=block_Y,loop=bk)

# load block get compute at the consumer loop
sch.compute_at(B_local_block,tk,preserve_unit_loops=False)
sch.compute_at(A_local_block,tk,preserve_unit_loops=False)
sch.compute_at(A_shared_block,bk,preserve_unit_loops=False)
sch.compute_at(B_shared_block,bk,preserve_unit_loops=False)
sch.reverse_compute_at(C_local_block,tx,preserve_unit_loops=False)

# add load stage before the share block
sch.compute_at(A_load_block,bk,preserve_unit_loops=False)
sch.compute_at(B_load_block,bk,preserve_unit_loops=False)

# cooperate fetch for share
ax0 = sch.get_loops(block=A_shared_block)[-2]
ax1 = sch.get_loops(block=A_shared_block)[-1]
tid = sch.fuse(ax0,ax1)
s_ty , s_tx , sh_iter = sch.split(tid,[16,16,4])
sch.bind(s_ty,"threadIdx.y")
sch.bind(s_tx,"threadIdx.x")
ax0 = sch.get_loops(block=B_shared_block)[-2]
ax1 = sch.get_loops(block=B_shared_block)[-1]
tid = sch.fuse(ax0,ax1)
s_ty , s_tx , sh_iter = sch.split(tid,[16,16,4])
sch.bind(s_ty,"threadIdx.y")
sch.bind(s_tx,"threadIdx.x")

# cooperate fetch for load
ax0 = sch.get_loops(block=A_load_block)[-2]
ax1 = sch.get_loops(block=A_load_block)[-1]
tid = sch.fuse(ax0,ax1)
s_ty , s_tx , sh_iter = sch.split(tid,[16,16,4])
sch.bind(s_ty,"threadIdx.y")
sch.bind(s_tx,"threadIdx.x")
ax0 = sch.get_loops(block=B_load_block)[-2]
ax1 = sch.get_loops(block=B_load_block)[-1]
tid = sch.fuse(ax0,ax1)
s_ty , s_tx , sh_iter = sch.split(tid,[16,16,4])
sch.bind(s_ty,"threadIdx.y")
sch.bind(s_tx,"threadIdx.x")

# break local loop
ax0 = sch.get_loops(block=A_local_block)[-1]
i , j = sch.split(loop=ax0,factors=[2,4])
ax0 = sch.get_loops(block=B_local_block)[-1]
i , j = sch.split(loop=ax0,factors=[2,4])

# avoid bank conflict
sch.transform_layout(block=B_local_block,buffer=("read",0),index_map= lambda i,j:( i , ((j//128)*128+(j%128)//8*4+(j%8)//4*64+j%4)))
sch.transform_layout(block=A_local_block,buffer=("read",0),index_map= lambda j,i:( j , ((i//128)*128+(i%128)//8*4+(i%8)//4*64+i%4)))

# unroll
ax0 = sch.get_loops(block=A_local_block)[-1]
ax1 = sch.get_loops(block=A_local_block)[-2]
ax2 = sch.get_loops(block=A_local_block)[-3]
sch.unroll(ax0)
sch.unroll(ax1)
sch.unroll(ax2)
ax0 = sch.get_loops(block=B_local_block)[-1]
ax1 = sch.get_loops(block=B_local_block)[-2]
sch.unroll(ax0)
sch.unroll(ax1)

# vectorize
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

# pipeline
sch.annotate(block_or_loop=A_shared_block,ann_key="double_buffer_scope",ann_val=0)
sch.annotate(block_or_loop=B_shared_block,ann_key="double_buffer_scope",ann_val=0)
ax0 = sch.get_loops(block=A_shared_block)[-4]
sch.annotate(block_or_loop=ax0,ann_key="software_pipeline_stage",ann_val=[0,0,0,0,1])
sch.annotate(block_or_loop=ax0,ann_key="software_pipeline_order",ann_val=[0,3,1,4,2])

# print(sch.mod.script())

rt_mod = tvm.build(sch.mod,target="cuda")
print(rt_mod.imported_modules[0].get_source())
