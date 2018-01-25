CXX=g++
RM=rm -f
CXXFLAGS=-O3 -I/usr/include/python3.5m/ -I. -fPIC
LDFLAGS=-O3 -shared
LDLIBS=-lboost_python-py35 -lboost_numpy3 -L/home/byrdie/School/Research/KSO/lib-kso/src/kso/cpp/

KSO_MK_INC = img/img.mk
include $(KSO_MK_INC)



all: $(KSO_ARTS)
	

clean: 
	$(RM) $(KSO_OBJS) $(KSO_ARTS) 
