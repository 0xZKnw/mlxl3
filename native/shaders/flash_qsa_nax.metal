// Cooperative 16x32x16 FP32 score tile, matching the M5 NAX matmul used by the reference.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

kernel void qsa_index_scores_nax(const device half* q [[buffer(0)]],const device half* blocks [[buffer(1)]],
    device float* out [[buffer(2)]],constant uint& count [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    constexpr auto desc=matmul2d_descriptor(16,32,16,false,false,true,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroup> op;
    auto a=op.get_left_input_cooperative_tensor<float,float,float>();
    auto b=op.get_right_input_cooperative_tensor<float,float,float>();
    auto c=op.get_destination_cooperative_tensor<decltype(a),decltype(b),float>();
    for (ushort i=0;i<c.get_capacity();++i) c[i]=0.0f;
    for(uint k=0;k<128;k+=16) {
        for(ushort i=0;i<a.get_capacity();++i) {
            auto ij=a.get_multidimensional_index(i);
            a[i]=ij[1]<4?float(q[uint(ij[1])*128+k+uint(ij[0])]):0.0f;
        }
        for(ushort i=0;i<b.get_capacity();++i) {
            auto ij=b.get_multidimensional_index(i);
            uint id=group*32+uint(ij[0]);b[i]=id<count?float(blocks[id*128+k+uint(ij[1])]):0.0f;
        }
        op.run(a,b,c);
    }
    threadgroup float scores[4*32];
    for(ushort i=0;i<c.get_capacity();++i) {
        auto ij=c.get_multidimensional_index(i);
        if(ij[1]<4) scores[uint(ij[1])*32+uint(ij[0])]=c[i];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    uint id=group*32+lane;
    if(id<count) {
        float sum=0;
        for(uint h=0;h<4;++h) sum+=max(scores[h*32+lane],0.0f);
        out[id]=sum*0.08838834764831844f;
    }
}
