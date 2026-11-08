#ifndef MLXL3_GDN_HISTORY
#define MLXL3_GDN_HISTORY(t,c) history[(t)*10240+(c)]
#define MLXL3_GDN_DEFAULT_HISTORY 1
#endif

    const uint sg=lid/32,ln=lid%32,hk=h/3;
    threadgroup half tq[128],tk[128],tv[128],ty[128];
    if(sg<3) {
        const uint channel_head=sg==0?hk:(sg==1?16+hk:32+h);
        half values[4],sum=0.0h;
        for(uint i=0;i<4;++i) {
            uint channel=channel_head*128+ln*4+i;
            float acc=0.0f;
            for(uint t=0;t<4;++t) {
                half x=t<3?MLXL3_GDN_HISTORY(t,channel):qkv[channel];
                acc+=float(x)*float(weight[channel*4+t]);
            }
            half x=half(acc);
            half tail=half(1.0h)/(half(1.0h)+metal::exp(metal::abs(x)));
            half sigmoid=x<half(0.0h)?tail:half(1.0h)-tail;
            half value=x*sigmoid;
            values[i]=value;
            volatile half squared=value*value;
            sum+=squared;
            // Shared Q/K heads have exactly one history owner; each V head is unique.
            if(sg==2 || h%3==0) for(uint t=0;t<3;++t)
                history_out[t*10240+channel]=t<2?MLXL3_GDN_HISTORY(t+1,channel):qkv[channel];
            if(sg==2)tv[ln*4+i]=value;
        }
        if(sg<2) {
            half total=simd_sum(sum);
            half inverse=metal::precise::rsqrt(half(total+half(1.0e-6f)));
            for(uint i=0;i<4;++i) {
                volatile half normalized=values[i]*inverse;
                if(sg==0)tq[ln*4+i]=normalized*half(0.08838834764831845f);
                else tk[ln*4+i]=normalized;
            }
        }
    }

#if MLXL3_GDN_HEAD_GATES
    threadgroup float shared_decay;
    threadgroup half shared_beta;
    if(sg==3 && ln==0) {
        half bh=b[h];
        half sigmoid_tail=half(1.0h)/(half(1.0h)+metal::exp(metal::abs(bh)));
        shared_beta=bh<half(0.0h)?sigmoid_tail:half(1.0h)-sigmoid_tail;
        half summed=half(float(a[h])+float(dt_bias[h]));
        half maximum=metal::max(summed,half(0.0h));
        half minimum=metal::min(summed,half(0.0h));
        float exp_delta=float(metal::exp(minimum-maximum));
        float plus_one=1.0f+exp_delta;
        float log_one_plus=plus_one==1.0f?exp_delta:exp_delta*(metal::log(plus_one)/(plus_one-1.0f));
        half softplus=half(float(maximum)+log_one_plus);
        shared_decay=precise::exp(-precise::exp(a_log[h])*float(softplus));
    }
#endif
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup half* q=tq;
    threadgroup half* k=tk;
    threadgroup half* v=tv;
    threadgroup half* y=ty;
    uint thread_index_in_simdgroup=ln;
    uint3 thread_position_in_grid=uint3(ln,sg,h);
#define MLXL3_GDN_LOCAL_QKV 1
#define MLXL3_GDN_FUSED_GATES 1
#define MLXL3_GDN_SHARED_GATES MLXL3_GDN_HEAD_GATES
#define MLXL3_GDN_SAVE_HISTORY 0
#define MLXL3_GDN_SAVE_TAPE 0
#define InT half
#define StT float
#define B 1
#define T 1
#define Dk 128
#define Dv 128
#define Hk 16
#define Hv 48
#include "gated_delta_packed.metal"
#undef MLXL3_GDN_LOCAL_QKV
#undef MLXL3_GDN_SHARED_GATES
#undef MLXL3_GDN_FUSED_GATES
#undef MLXL3_GDN_SAVE_HISTORY
#undef MLXL3_GDN_SAVE_TAPE
#undef InT
#undef StT
#undef B
#undef T
#undef Dk
#undef Dv
#undef Hk
#undef Hv
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sg==0) {
        const uint at=h*128+ln*4;
        float values[4],sum=0;
        for(uint i=0;i<4;++i){values[i]=float(ty[ln*4+i]);sum+=values[i]*values[i];}
        float inverse=precise::rsqrt(simd_sum(sum)/128.0f+as_type<float>(params[0]));
        for(uint i=0;i<4;++i) {
            volatile float normalized=gain[ln*4+i]*(values[i]*inverse);
            float gate=float(z[at+i]);
            float p=1.0f/(1.0f+metal::precise::exp(metal::abs(gate)));
            volatile float probability=gate<0.0f?p:1.0f-p;
            volatile float product=normalized*probability;
            gated[at+i]=half(product);
        }
    }

#ifdef MLXL3_GDN_DEFAULT_HISTORY
#undef MLXL3_GDN_HISTORY
#undef MLXL3_GDN_DEFAULT_HISTORY
#endif
