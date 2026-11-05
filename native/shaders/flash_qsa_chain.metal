// Canonical stripe recurrence; only independent key/value loads may run ahead.
    float q[AR20_HPT][8], o[AR20_HPT][8], top[AR20_HPT], sum[AR20_HPT];
    for(uint h=0;h<AR20_HPT;++h) {
        top[h]=-FLT_MAX; sum[h]=0;
        for(uint j=0;j<8;++j) {q[h][j]=float(queries[(first_head+h)*256+lane*8+j])/16.0f;o[h][j]=0;}
    }
#if AR20_PREFETCH
    half ks[4][8],vs[4][8];
    for(uint slot=0;slot<4;++slot) {
        uint t=chain+slot*32;
        if(t<params[0]) for(uint j=0;j<8;++j) {uint base=t*512+(first_head/12)*256+lane*8;ks[slot][j]=keys[base+j];vs[slot][j]=values[base+j];}
    }
    for(uint t=chain;t<params[0];t+=128) {
        float score[AR20_HPT][4];
        for(uint slot=0;slot<4;++slot) if(t+slot*32<params[0]) {
            for(uint h=0;h<AR20_HPT;++h) {
                float acc=0;
                for(uint j=0;j<8;++j) acc+=q[h][j]*float(ks[slot][j]);
                score[h][slot]=simd_sum(acc);
            }
        }
        // Next K may issue before the current online state update.
        for(uint slot=0;slot<4;++slot) if(t+(slot+4)*32<params[0]) {
            uint base=(t+(slot+4)*32)*512+(first_head/12)*256+lane*8;
            for(uint j=0;j<8;++j) ks[slot][j]=keys[base+j];
        }
        for(uint slot=0;slot<4;++slot) {
            if(t+slot*32>=params[0]) break;
            for(uint h=0;h<AR20_HPT;++h) {
                float next=max(top[h],score[h][slot]),factor=fast::exp(top[h]-next),weight=fast::exp(score[h][slot]-next);
                top[h]=next;sum[h]=sum[h]*factor+weight;
                for(uint j=0;j<8;++j) o[h][j]=o[h][j]*factor+weight*float(vs[slot][j]);
            }
            if(t+(slot+4)*32<params[0]) {
                uint base=(t+(slot+4)*32)*512+(first_head/12)*256+lane*8;
                for(uint j=0;j<8;++j) vs[slot][j]=values[base+j];
            }
        }
    }
#else
    for(uint t=chain;t<params[0];t+=32) {
        uint base=t*512+(first_head/12)*256+lane*8;
        half ks[8],vs[8];
        for(uint j=0;j<8;++j) {ks[j]=keys[base+j];vs[j]=values[base+j];}
        for(uint h=0;h<AR20_HPT;++h) {
            float score=0;
            for(uint j=0;j<8;++j) score+=q[h][j]*float(ks[j]);
            score=simd_sum(score);
            float next=max(top[h],score),factor=fast::exp(top[h]-next),weight=fast::exp(score-next);
            top[h]=next;sum[h]=sum[h]*factor+weight;
            for(uint j=0;j<8;++j) o[h][j]=o[h][j]*factor+weight*float(vs[j]);
        }
    }
#endif
