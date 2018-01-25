COMMA = ,

CXX=g++
RM=rm -f
CXXFLAGS=-O3 -I/usr/include/python3.5m/ -I. -fPIC
LDFLAGS=-O3 -shared
LDLIBS=-lboost_python-py35 -lboost_numpy3 -L.

RP_ORIG=$$ORIGIN

KSO_MK_INC = img/img.mk
KSO_MK_INC += util/util.mk
include $(KSO_MK_INC)



all: $(KSO_ARTS)

clean: 
	$(RM) $(KSO_OBJS) $(KSO_ARTS) 
