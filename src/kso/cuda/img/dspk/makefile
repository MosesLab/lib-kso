NVCC=/usr/local/cuda-8.0/bin/nvcc
RM=rm -f
CUDAFLAGS=-O3 -I/usr/include/python3.5m/ --cudart static --relocatable-device-code=false -gencode arch=compute_52,code=compute_52 -gencode arch=compute_52,code=sm_52 --compiler-options '-fPIC'
LDFLAGS=-O3 -shared -Wno-deprecated-gpu-targets
LDLIBS=-lboost_python-py35 -lboost_numpy3

SRCS=dspk_cuda.cu
HDRS=dspk_cuda.h
OBJS=dspk_cuda.o
ARTS=dspk_cuda.so


all: $(ARTS)

dspk_cuda.so: $(OBJS) 
	$(NVCC) $(LDFLAGS) -o dspk_cuda.so $(OBJS) $(LDLIBS) 

dspk_cuda.o: $(SRCS) $(HDRS)
	$(NVCC) $(CUDAFLAGS) -c $(SRCS) -o dspk_cuda.o

clean: 
	$(RM) $(OBJS) $(ARTS) 
