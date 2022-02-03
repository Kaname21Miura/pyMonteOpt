#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 17:15:45 2021
@author: Kaname Miura
"""
import cupy as cp

def vmc_kernel():
    func = cp.RawKernel(r"""
    #include <curand_kernel.h>

    extern "C"{
        __device__ float theta(float g, float rand) {
            float th = 0;
            if (g != 0.) {
                float g2 = powf(g, 2);
                th = (1 + g2 - powf(((1 - g2) / (1 - g + 2 * g * rand)), 2)) / (2 * g);
                if (th < -1){th = -1;}
            }
            else {
                th = 2 * rand - 1;
            }
            return th;
        }

        __global__ void cuVMC(
            volatile int* add,
            volatile float* p,
            volatile float* v,
            volatile float* w,
            float* ma, float* ms, float* n, float* g,
            char* voxel_model, float l,
            int M, int L, int nPh, char end_point, int rand_seed
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
                float zero_vec[3] = {0,0,0}, one_vec[3] = { 1,1,1 };
                float fi = 0;
                float cos_fi, cos_th;
                float sin_fi, sin_th;

                const float wth = 0.0001;
                const float roulette_m = 10;

                char index = voxel_model[add[ix] * M * L + add[iy] * L + add[iz]];
                char index_next = 0;
                const char index_end = end_point;

                float ni = n[index];
                float nt = 0;
                float mt = ma[index] + ms[index];

                float st = 0;

                float valf, db;
                int dbnum, dbid;
                float ai = 0, at = 0, ra = 0;

                bool flag = 1;
                //int step = 0;

                while (flag) {
                    // get photon’s step size
                    st = -logf(curand_uniform(&s));

                    while (1) {
                        // The distance between the current photon location
                        // and the boundary of the current voxel
                        valf = 0, dbnum = 0, dbid = 0, db = 1000;
                        for (int i = 0; i < 3; i++) {
                            if (fabsf(v[idx + nPh * i]) > 0) {
                                valf = (l / 2 - copysignf(1, v[idx + nPh * i]) * p[idx + nPh * i])
                                / fabsf(v[idx + nPh * i]);
                                if (valf < db) {
                                    dbnum = i, db = valf;
                                }
                            }
                        }
                        dbid = idx + nPh * dbnum;

                        if (st >= db * mt) {// １ステップがvoxel境界を越える場合
                            // 光子を境界まで移動
                            for (int i = 0; i < 3; i++) { p[idx + nPh * i] += v[idx + nPh * i] * db; }
                            p[dbid] = copysignf(l / 2, v[dbid]);// 計算誤差を補正
                            st -= db * mt;

                            // 透過先の光学特性indexを入手
                            for (int i = 0; i < 3; i++) { add_[i] = add[idx + nPh * i]; }//add_の初期化
                            add_[dbnum] += (int)copysignf(1, v[dbid]);
                            index_next = voxel_model[add_[0] * M * L + add_[1] * L + add_[2]];

                            // 透過先の屈折率を入手
                            nt = n[index_next];
                            if (ni != nt) {// 屈折率が変化した場合
                                // 透過判別
                                ai = acosf(fabsf(v[dbid])); // 入射角
                                if ((ni > nt) && (ai >= asinf(nt / ni))) {// 全反射する場合
                                    ra = 1.;
                                }
                                else {// 全反射しない場合
                                    at = asinf((ni/nt) * sinf(ai));
                                    if(ai!=0){// 非常に重要
                                    // ai = 0の場合このままだとraはnanになる
                                    ra = (powf((sinf(ai - at) / sinf(ai + at)), 2)
                                    + powf((tanf(ai - at) / tanf(ai + at)), 2))/2;
                                    }
                                    else{
                                    ra = 0.;
                                    }
                                }
                                valf = curand_uniform(&s);

                                if (ra < valf) { //透過
                                    // ベクトルを更新
                                    zero_vec[dbnum] = 1, one_vec[dbnum] = 0;
                                    for (int i = 0; i < 3; i++) {
                                        v[idx + nPh * i] = one_vec[i] * v[idx + nPh * i] * ni / nt
                                            + zero_vec[i] * copysignf(1, v[idx + nPh * i]) * cosf(at);
                                        zero_vec[i] = 0, one_vec[i] = 1;
                                    }
                                    valf = 0;
                                    for (int i = 0; i < 3; i++) { valf += powf(v[idx + nPh * i],2); }
                                    for (int i = 0; i < 3; i++) { v[idx + nPh * i] /= sqrtf(valf); }

                                    // Addressを更新
                                    add[dbid] += (int)copysignf(1, v[dbid]);

                                    // Voxel内の位置を更新
                                    p[dbid] *= -1;
                                    // 光学特性indexを更新
                                    index = index_next;
                                    if (index == index_end) {// 更新先が終端voxelだった場合
                                        goto End;
                                    }
                                    // 光学特性を更新
                                    mt = ma[index] + ms[index];
                                    ni = nt;
                                }
                                else { // Fresnel 反射
                                    v[dbid] *= -1; //ベクトルの更新
                                }
                            }
                            else {//透過するときの処理

                                // Addressを更新
                                add[dbid] += (int)copysignf(1, v[dbid]);

                                // Voxel内の位置を更新
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
                            for (int i = 0; i < 3; i++) { p[idx + nPh * i] += (v[idx + nPh * i] * st / mt); }
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
                                p[idx + nPh * i] = 0;
                                v[idx + nPh * i] = 0;
                                add[idx + nPh * i] = 0;
                            }
                            w[idx] = 0;
                            goto End;
                        }
                        else {
                            w[idx] *= roulette_m;
                        }
                    }
                    // Photon scattering
                    cos_th = theta(g[index], curand_uniform(&s));
                    sin_th = sqrtf(1-powf(cos_th,2));

                    fi = 2 * 3.1415927*curand_uniform(&s);
                    cos_fi = cosf(fi);
                    sin_fi = sinf(fi);

                    if (0.99999 < fabsf(v[iz])) {
                        v[ix] = sin_th * cos_fi;
                        v[iy] = sin_th * sin_fi;
                        v[iz] = copysignf(1, v[iz]) * cos_th;
                    }
                    else {
                        valf = sqrtf(1 - v[iz]*v[iz]);
                        v_[0] = sin_th * (v[ix] * v[iz] * cos_fi - v[iy] * sin_fi) / valf + v[ix] * cos_th;
                        v_[1] = sin_th * (v[iy] * v[iz] * cos_fi + v[ix] * sin_fi) / valf + v[iy] * cos_th;
                        v_[2] = -sin_th * cos_fi * valf + v[iz] * cos_th;
                        for (int i = 0; i < 3; i++){ v[idx + nPh * i] = v_[i]; }
                    }
                    // 計算誤差の補正（単位ベクトルに変換）
                    valf = 0;
                    for (int i = 0; i < 3; i++) { valf += powf(v[idx + nPh * i],2); }
                    for (int i = 0; i < 3; i++) { v[idx + nPh * i] /= sqrtf(valf); }
                    /*
                    if (step == 0){flag = 0;}
                    step += 1;
                    */
                }
            End:
            }
        }
    }
      """, "cuVMC")
    return func
