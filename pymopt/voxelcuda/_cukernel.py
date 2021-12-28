#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 17:15:45 2021
@author: Kaname Miura
"""
from pycuda.compiler import SourceModule

def vmc_kernel():
    
    mod = SourceModule("""
    #include <curand_kernel.h>

    extern "C"{
        __device__ float theta(float g_, float rand) {
            float th = 0;
            if (g_ != 0.) {
                float g2_ = powf(g_, 2);
                th = acosf((1 + g2_ - powf(((1 - g2_)/(1 - g_ + 2 * g_ * rand)), 2)) / (2 * g_));
            }
            else {
                th = acosf(2 * rand - 1);
            }
            return th;
        }
        __device__ float phi(float rand) {
            return 2 * 3.14159 * rand;
        }
        __device__ float get_at(float ai, float n0, float n1) {
            return asinf(n0 * sinf(ai) / n1);
        }
        __device__ float get_ra(float ai, float at) {
            return 0.5 * (powf((sinf(ai - at)/ sinf(ai + at)),2) + powf((tanf(ai - at)/ tanf(ai + at)),2));
        }
        __global__ void cuVMC(
            int* add,
            float* p,
            float* v,
            float* w,
            float* ma, float* ms, float* n, float* g,
            char* voxel_model, float l,
            int M, int L, int nPh, char end_point,int rand_seed
        ) {
            const int idx = threadIdx.x + blockDim.x * blockIdx.x;
            if (idx < nPh) {
                const int ix = idx;
                const int iy = idx + nPh;

                const int iz = idx + 2 * nPh;

                curandState s;
                curand_init(rand_seed, idx, 0, &s);

                // 変数系　計算用メモリ
                int add_[3] = {};
                float v_[3] = {};
                float sum_v = 0;
                float zero_vec[3] = {}, one_vec[3] = { 1,1,1 };
                float th = 0, fi = 0;
                float cos_fi,cos_th;
                float sin_fi,sin_th;

                const float wth = 0.0001;
                const float roulette_m = 10;

                char index = voxel_model[add[ix] * M * L + add[iy] * L + add[iz]];
                char index_next = 0;
                const char index_end = end_point;

                float n0 = n[index];
                float n1 = 0;
                float mt= ma[index] + ms[index];

                float st = 0;

                const float ilig_v = 0;
                float valf, db;
                int dbnum, dbid;

                float ai = 0, ai_max = 0, at = 0, ra = 0;

                bool flag = 1;
                //int step = 0;

                while (flag) {
                    // get photon’s step size
                    st = -logf(curand_uniform(&s));

                    while (1) {
                        // The distance between the current photon location
                        // and the boundary of the current voxel
                        valf = 0, dbnum = 0, dbid = 0, db = 100;
                        for (int i = 0; i < 3; i++) {
                            if (v[idx + nPh * i] != ilig_v) {
                                valf = (l / 2 - p[idx + nPh * i] * copysignf(1,v[idx + nPh * i]))
                                / fabsf(v[idx + nPh * i]);
                                if (valf < db) {
                                    dbnum = i, db = valf;
                                }
                            }
                        }
                        dbid = idx + nPh * dbnum;

                        if ((st - db * mt) > 0) {// １ステップがvoxel境界を越える場合

                            // 光子を境界まで移動
                            for (int i = 0; i < 3; i++) {p[idx + nPh * i] += v[idx + nPh * i] * db;}
                            p[dbid] = copysignf(l, v[dbid]) / 2;// 計算誤差を補正
                            st -= db * mt;

                            // 透過先のa光学特性indexを入手
                            for (int i = 0; i < 3; i ++){add_[i] = add[idx+nPh*i];}//add_の初期化
                            add_[dbnum] = add[dbid]+(int)copysignf(1, v[dbid]);
                            index_next = voxel_model[add_[0] * M * L + add_[1] * L + add_[2]];

                            // 透過先の屈折率を入手
                            n1 = n[index_next];

                            if (n0 != n1) {// 屈折率が変化した場合
                                // 透過判別
                                ai = acosf(fabsf(v[dbid])); // 入射角

                                if (n0 > n1) {// 全反射する可能性がある場合
                                    ra = 1; // 初期化(全反射)
                                    ai_max = asinf(n1 / n0);// critical angle
                                    if (ai < ai_max) { //全反射しない場合
                                        at = get_at(ai, n0, n1);
                                        ra = get_ra(ai, at);
                                    }
                                }
                                else {// 全反射する可能性がない場合
                                    at = get_at(ai, n0, n1);
                                    ra = get_ra(ai, at);
                                }
                                if (ra < curand_uniform(&s)) { //透過
                                    // Addressを更新
                                    add[dbid] += (int)copysignf(1, v[dbid]);

                                    // Address内の位置を更新
                                    p[dbid] *= -1;

                                    // ベクトルを更新
                                    zero_vec[dbnum] = 1, one_vec[dbnum] = 0;
                                    for (int i = 0; i < 3; i++) {
                                        v[idx + nPh * i] = zero_vec[i] * copysignf(1, v[idx + nPh * i]) * cosf(at)
                                            + one_vec[i] * v[idx + nPh * i] * n0 / n1;
                                    }
                                    for (int i = 0; i < 3; i++) {// 初期化
                                        zero_vec[i] = 0;
                                        one_vec[i] = 1;
                                    }
                                    // 光学特性を更新
                                    n0 = n1;
                                    index = index_next;
                                    if (index == index_end) {// 更新先が終端voxelだった場合
                                        goto End;
                                    }
                                    mt = ma[index] + ms[index];
                                }
                                else { // Fresnel 反射
                                    v[dbid] *= -1; //ベクトルの更新
                                }
                            }
                            else { // voxelを移動(透過)
                                // Addressを更新
                                add[dbid] += (int)copysignf(1, v[dbid]);

                                // Address内の位置を更新
                                p[dbid] *= -1;

                                // 光学特性indexを更新
                                index = index_next;

                                if (index == index_end) {// 更新先が終端voxelだった場合
                                    goto End;
                                }
                                // 光学特性を更新
                                mt = ma[index] + ms[index];
                            }
                        }
                        else {
                            // Pthoton moving
                            for (int i = 0; i < 3; i++) {p[idx + nPh * i] += v[idx + nPh * i] * st / mt;}
                            st = 0;
                            break;
                        }
                    }

                    // Photon absorption
                    w[idx] -= w[idx] * ma[index] / mt;

                    // serviving
                    if (w[idx] <= wth) {
                        if ((1 / roulette_m) < curand_uniform(&s)) {

                            for (int i = 0; i < 3; i++) {
                                p[idx + nPh * i] = NAN;
                                v[idx + nPh * i] = NAN;
                                add[idx + nPh * i] = NAN;
                            }
                            w[idx] = NAN;
                            goto End;
                        }
                        else {
                            w[idx] *= roulette_m;
                        }
                    }
                    // Photon scattering
                    th = theta(g[index], curand_uniform(&s));
                    fi = phi(curand_uniform(&s));
                    cos_fi = cosf(fi),cos_th = cosf(th);
                    sin_fi = sinf(fi),sin_th = sinf(th);
                    if (0.99999 <= v[iz]) {
                        v[ix] = sin_th * cos_fi;
                        v[iy] = sin_th * sinf(fi);
                        v[iz] = copysignf(1,v[iz]) * cos_th;
                    }
                    else {
                        valf = sqrtf(1 - powf(v[iz],2));
                        v_[0] = sin_th * (v[ix] * v[iz] * cos_fi - v[iy] * sin_fi) /valf + v[ix] * cos_th;
                        v_[1] = sin_th * (v[iy] * v[iz] * cos_fi + v[ix] * sin_fi) /valf + v[iy] * cos_th;
                        v_[2] = -sin_th * cos_fi * valf + v[iz] * cos_th;
                        for(int i = 0; i < 3; i++){v[idx + nPh * i] = v_[i];}
                        valf = 0;
                    }
                    // 計算誤差の補正（単位ベクトルに変換）
                    sum_v = 0;
                    for (int i = 0; i < 3; i++) {sum_v += powf(v[idx + nPh * i],2);}
                    for (int i = 0; i < 3; i++) {v[idx + nPh * i] /= sqrtf(sum_v);}

                    /*
                    if (step == 1){
                        flag = 0;
                    }
                    step += 1;
                    */
                }
            }
            End:

        }
    }
      """, no_extern_c=True)
    return mod
