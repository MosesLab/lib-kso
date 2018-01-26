/*
 * math.cpp
 *
 *  Created on: Jan 25, 2018
 *      Author: byrdie
 */

#include "math.h"

namespace kso {

namespace math {

void add(float * out, float * a, float * b, ku::dim3 sz){

	// retrieve sizes
	uint sz_x = sz.x;
	uint sz_y = sz.y;
	uint sz_z = sz.z;

	uint L = sz.x * sz.y * sz.z;

	for(uint i = 0; i < L; i++){
		out[i] = a[i] + b[i];
	}


}
void sub(float * out, float * a, float * b, ku::dim3 sz){
	// retrieve sizes
	uint sz_x = sz.x;
	uint sz_y = sz.y;
	uint sz_z = sz.z;

	uint L = sz.x * sz.y * sz.z;

	for(uint i = 0; i < L; i++){
		out[i] = a[i] - b[i];
	}
}
void mul(float * out, float * a, float * b, ku::dim3 sz){
	// retrieve sizes
	uint sz_x = sz.x;
	uint sz_y = sz.y;
	uint sz_z = sz.z;

	uint L = sz.x * sz.y * sz.z;

	for(uint i = 0; i < L; i++){
		out[i] = a[i] * b[i];
	}
}
void div(float * out, float * a, float * b, ku::dim3 sz){
	// retrieve sizes
	uint sz_x = sz.x;
	uint sz_y = sz.y;
	uint sz_z = sz.z;

	uint L = sz.x * sz.y * sz.z;

	for(uint i = 0; i < L; i++){
		out[i] = a[i] / b[i];
	}
}

}

}

