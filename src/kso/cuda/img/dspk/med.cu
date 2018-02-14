
#include "med.h"

namespace kso {

namespace img {

namespace dspk {

using namespace std;

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
				break;
			}
		}

	}





}

}

}

}
