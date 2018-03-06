
#include "med.h"

namespace kso {

namespace img {

namespace dspk {

using namespace std;

__global__ void calc_gm(float * gm, uint * new_bad, float * dt, float * q2, float * t0, float * t1, dim3 sz, dim3 hsz){

	// retrieve coordinates from thread and block id.
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;

	// compute stride sizes
	dim3 n;
	n.x = 1;
	n.y = n.x * sz.x;
	n.z = n.y * sz.y;


	// compute histogram strides
	dim3 m;
	m.x = 1;
	m.y = m.x * hsz.x;
	m.z = 0;

	// overall index
	uint L = n.z * z + n.y * y + n.x * x;

	if (gm[L] == 0.0) return;

	// calculate width of histogram bins
	dim3 bw;
	bw.x = Dt / (hsz.x - 1);
	bw.y = Dt / (hsz.y - 1);
	bw.z = 0;


	float dt_0 = dt[L];

	// load median and intensity at this point
	float q2_0 = q2[L];


	// calculate histogram indices
	uint X = (q2_0 - dt_min) / bw.x;
	uint Y = (dt_0 - dt_min) / bw.y;

	if((((uint) t0[X]) >= Y) or (((uint) t1[X]) <= Y)){

		gm[L] = 0.0f;
		dt[L] = 0.0f;
		atomicAdd(new_bad, 1);

	}

}




__global__ void calc_gm(float * gm, uint * new_bad, float * dt, float * q2, float * t0, float * t1, dim3 sz, dim3 hsz, uint nmet){

	// retrieve coordinates from thread and block id.
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;

	// compute stride sizes
	dim3 n;
	n.x = 1;
	n.y = n.x * sz.x;
	n.z = n.y * sz.y;

	uint sz3 = sz.x * sz.y * sz.z;

	// compute histogram strides
	dim3 m;
	m.x = 1;
	m.y = m.x * hsz.x;
	m.z = 0;

	// overall index
	uint L = n.z * z + n.y * y + n.x * x;

	if (gm[L] == 0.0) return;

	// calculate width of histogram bins
	dim3 bw;
	bw.x = Dt / (hsz.x - 1);
	bw.y = Dt / (hsz.y - 1);
	bw.z = 0;

	uint wgt[4] = {1, 1, 1, 1};
	uint tot_wgt = 0;

	uint votes = 0;

	float dt_0 = dt[L];

	for(uint ax = 0; ax < nmet; ax++){

		uint Lx = L + ax * sz3;

		// load median and intensity at this point
		float q2_0 = q2[Lx];




		// calculate histogram indices
		uint X = (q2_0 - dt_min) / bw.x;
		uint Y = (dt_0 - dt_min) / bw.y;

		if((((uint) t0[X]) >= Y) or (((uint) t1[X]) <= Y)){

			votes += wgt[ax];

		}
		tot_wgt += wgt[ax];

	}

	if(votes >= (tot_wgt)){

		gm[L] = 0.0f;
				dt[L] = 0.0f;
				atomicAdd(new_bad, 1);

	}




}

__global__ void calc_thresh(float * t0, float * t1, float * hist, float * cs, dim3 hsz, float T0, float T1){

	// retrieve index
	uint i = threadIdx.x;

	// compute histogram strides
	dim3 m;
	m.x = 1;
	m.y = m.x * hsz.x;
	m.z = 0;

	// march along y and find value of threshold
	bool f1 = false;
	int j;
	for(j = 0; j < hsz.y; j++){

		// linear index in histogram
		uint M = m.x * i + m.y * j;

		if(f1 == false){
			if(cs[M] >= T0){

				f1 = true;
				t0[i] = max(j - 1,0);

			}
		} else {

			if(cs[M] >= T1){

				t1[i] = j + 1;
				return;

			}

		}


	}
	uint slope = hsz.y / hsz.x;
	t0[i] = slope * i;
	t1[i] = slope * i;


}

__global__ void smooth_thresh(float * out, float * in, dim3 hsz, uint kern_sz){

	uint i = threadIdx.x;


	// initialize mean
	float mean = 0.0;
	float norm = 0.0;

	uint ks2 = kern_sz / 2;

	// convolve
	for(uint x = 0; x < kern_sz; x++){

		// calculate offset
		uint X = i - ks2 + x;

		// truncate kernel if we're over the edge
		if(X > (hsz.x - 1)){
			continue;
		}


		mean = mean + in[X];
		norm = norm + 1.0f;

	}

	mean = mean / norm;

	out[i] = mean;

}

__global__ void calc_cumsum(float * cs, float * hist, dim3 hsz){

	// retrieve index
	uint i = threadIdx.x;

	// compute histogram strides
	dim3 m;
	m.x = 1;
	m.y = m.x * hsz.x;
	m.z = 0;

	// march along y to build cumulative distribution
	float sum = 0.0f;
	for(uint j = 0; j < hsz.y; j++){

		// linear index in histogram
		uint M = m.x * i + m.y * j;

		// increment sum
		sum = sum + hist[M];

		// store result
		cs[M] = sum;

	}

	// renormalize
	for(uint j = 0; j < hsz.y; j++){

		// linear index in histogram
		uint M = m.x * i + m.y * j;

		cs[M] = cs[M] / sum;

		if(sum != 0.0f){
			hist[M] = hist[M] / sum;
		} else {
			hist[M] = 0.0f;
		}

	}



}

__global__ void calc_hist(float * hist, float * dt, float * q2, float * gm, dim3 sz, dim3 hsz){

	// retrieve coordinates from thread and block id.
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;

	// compute stride sizes
	dim3 n;
	n.x = 1;
	n.y = n.x * sz.x;
	n.z = n.y * sz.y;

	// compute histogram strides
	dim3 m;
	m.x = 1;
	m.y = m.x * hsz.x;
	m.z = 0;

	// overall index
	uint L = n.z * z + n.y * y + n.x * x;

	if (gm[L] == 0.0) return;	// don't do anything if the goodmap marks this pixel as bad

	// load median and intensity at this point
	float dt_0 = dt[L];
	float q2_0 = q2[L];


	//	if(Dq < hsz.x){			// make sure the median doesn't have too many bins
	//		hsz.x = Dq;
	//	}

	// calculate width of histogram bins
	dim3 bw;
	bw.x = Dt / (hsz.x - 1);
	bw.y = Dt / (hsz.y - 1);
	bw.z = 0;

	// calculate histogram indices
	uint X = (q2_0 - dt_min) / bw.x;
	uint Y = (dt_0 - dt_min) / bw.y;
	uint M = m.x * X + m.y * Y;



	//	uint Y_0 = (0 - dt_min) / bw.y;

	// update histogram
	atomicAdd(hist + M, 1);
}

void calc_quartile(float * q, float * dt, float * gm, float * tmp, dim3 sz, dim3 ksz, uint quartile){


	const dim3 xhat(1,0,0);
	const dim3 yhat(0,1,0);
	const dim3 zhat(0,0,1);

	dim3 threads(sz.x,1,1);
	dim3 blocks(1, sz.y, sz.z);


	calc_sep_quartile<<<blocks, threads>>>(q, dt, gm, sz, ksz, xhat, quartile);
	calc_sep_quartile<<<blocks, threads>>>(tmp, q, gm, sz, ksz, yhat, quartile);
	calc_sep_quartile<<<blocks, threads>>>(q, tmp, gm, sz, ksz, zhat, quartile);
	//	calc_tot_quartile<<<blocks, threads>>>(q, dt, gm, sz, ksz, quartile);


}

void calc_quartiles(float * q1, float * q2, float * q3, float * dt, float * gm, float * tmp, dim3 sz, dim3 ksz){

	vector<float * > q;
	q.push_back(q1);
	q.push_back(q2);
	q.push_back(q3);

	const dim3 xhat(1,0,0);
	const dim3 yhat(0,1,0);
	const dim3 zhat(0,0,1);

	dim3 threads(sz.x,1,1);
	dim3 blocks(1, sz.y, sz.z);


	// loop over each quartile
	for(uint quartile = 1; quartile <=3; quartile++){

		float * Q = q[quartile - 1];

		calc_sep_quartile<<<blocks, threads>>>(Q, dt, gm, sz, ksz, xhat, quartile);
		calc_sep_quartile<<<blocks, threads>>>(tmp, Q, gm, sz, ksz, yhat, quartile);
		calc_sep_quartile<<<blocks, threads>>>(Q, tmp, gm, sz, ksz, zhat, quartile);

	}

}

__global__ void calc_sep_quartile(float * q_out, float * q_in, float * gm, dim3 sz, dim3 ksz, dim3 axis, uint quartile){


	dim3 a = axis;

	// calculate offset for kernel
	dim3 kr;
	kr.x = ksz.x / 2;
	kr.y = ksz.y / 2;
	kr.z = ksz.z / 2;

	// compute stride sizes
	dim3 n;
	n.x = 1;
	n.y = n.x * sz.x;
	n.z = n.y * sz.y;


	// separate out quartile arrays
	float * q = q_in;


	// retrieve coordinates from thread and block id.
	uint x = blockIdx.x * blockDim.x + threadIdx.x;
	uint y = blockIdx.y * blockDim.y + threadIdx.y;
	uint z = blockIdx.z * blockDim.z + threadIdx.z;


	// overall index
	uint L = n.z * z + n.y * y + n.x * x;

	// size of the kernel along the selected axis
	uint kr1 = (kr.x * a.x) + (kr.y * a.y) + (kr.z * a.z);

	// size along selected axis
	uint sz1 = (sz.x * a.x) + (sz.y * a.y) + (sz.z * a.z);

	// index along current axis
	uint r = (x * a.x) + (y * a.y) + (z * a.z);

	// outer loop (select value)
	for(int i = -((int)kr1); i <= (int)kr1; i++){

		// Check if inside bounds
		uint I = r + i;
		if(I > (sz1 - 1)){
			continue;
		}

		// select possible quartile values
		uint A = n.z * (z + a.z * i) + n.y * (y + a.y * i) + n.x * (x + a.x * i);
		float u = q[A];


		// initialize memory for each bin
		int sm = 0;
		int eq = 0;
		int lg = 0;

		// inner loop (compare to other values)
		for(int j = -((int)kr1); j <= (int)kr1; j++){

			// check if inside bounds
			uint J = r + j;
			if(J > (sz1 - 1)){
				continue;
			}

			// select comparison values
			uint B = n.z * (z + a.z * j) + n.y * (y + a.y * j) + n.x * (x + a.x * j);
			float v = q[B];

			if(gm[B] == 0.0f){
				continue;
			}

			// increment counts for each bin
			if(u > v) {
				sm = sm + 1;
			} else if (u == v){
				eq = eq + 1;;
			} else {
				lg = lg + 1;
			}


		}

		// total count
		uint tot = sm + eq + lg;

		// quartile indices
		int P = quartile * tot / 4;



		// check quartile
		if((sm - 1) < P){
			if((sm + eq - 1) >= P){
				q_out[L] = u;
				return;
			}
		}

	}
	q_out[L] = 0.0f;

}

__global__ void calc_tot_quartile(float * q, float * dt, float * gm, dim3 sz, dim3 ksz, uint quartile){

	// calculate offset for kernel
	dim3 kr;
	kr.x = ksz.x / 2;
	kr.y = ksz.y / 2;
	kr.z = ksz.z / 2;

	// compute stride sizes
	dim3 n;
	n.x = 1;
	n.y = n.x * sz.x;
	n.z = n.y * sz.y;


	// retrieve coordinates from thread and block id.
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;


	// overall index
	uint L = n.z * z + n.y * y + n.x * x;


	//------------------------------------------------
	// select value
	for(int K = -((int)kr.z); K <= (int)kr.z; K++){
		int C = z + K;
		if (C > sz.z - 1) continue;

		for(int J = -((int)kr.y); J <= (int)kr.y; J++){
			int B = y + J;
			if (B > sz.y - 1) continue;

			for(int I = -((int)kr.x); I <= (int)kr.x; I++){
				int A = x + I;
				if (A > sz.x - 1) continue;

				int M = n.z * C + n.y * B + n.x * A;
				float u = dt[M];

				// initialize memory for each bin
				int sm = 0;
				int eq = 0;
				int lg = 0;

				// --------------------------------------------------------------
				// comparison
				for(int k = -((int)kr.z); k <= (int)kr.z; k++){
					int c = z + k;
					if (c > sz.z - 1) continue;

					for(int j = -((int)kr.y); j <= (int)kr.y; j++){
						int b = y + j;
						if (b > sz.y - 1) continue;

						for(int i = -((int)kr.x); i <= (int)kr.x; i++){
							int a = x + i;
							if (a > sz.x - 1) continue;

							int m = n.z * c + n.y * b + n.x * a;
							float v = dt[m];

							if(gm[m] == 0.0f) continue;

							// increment counts for each bin
							if(u > v) {
								sm = sm + 1;
							} else if (u == v){
								eq = eq + 1;;
							} else {
								lg = lg + 1;
							}


						}
					}
				}

				// total count
				int tot = sm + eq + lg;

				// quartile indices
				int P = quartile * tot / 4;



				// check quartile
				if((sm - 1) < P){
					if((sm + eq - 1) >= P){
						q[L] = u;
						break;
					}
				}

			}

		}

	}




}

__global__ void init_hist(float * hist, dim3 hsz, uint nmet){

	uint i = threadIdx.x;
	uint j = blockIdx.x;

	dim3 m;
	m.x = 1;
	m.y = m.x * hsz.x;
	m.z = 0;

	uint hsz3 = hsz.x * hsz.y;

	uint L = i * m.x + j * m.y;

	for(uint ax = 0; ax < nmet; ax++){

		hist[L + ax * hsz3] = 0;

	}



}

__global__ void init_thresh(float * t0, float * t1, dim3 hsz, uint nmet){

	uint i = threadIdx.x;

	dim3 m;
	m.x = 1;
	m.y = m.x * hsz.x;
	m.z = 0;


	for(uint ax = 0; ax < nmet; ax++){

		t0[i + ax * hsz.x] = 0.0f;
		t1[i + ax * hsz.x] = 0.0f;

	}



}

}

}

}
