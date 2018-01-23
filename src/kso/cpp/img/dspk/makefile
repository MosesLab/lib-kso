CXX=g++
RM=rm -f
CXXFLAGS=-O3 -I/usr/include/python3.5m/ -fPIC
LDFLAGS=-O3 -shared
LDLIBS=-lboost_python-py35 -lboost_numpy3

SRCS=dspk.cpp
HDRS=dspk.h
OBJS=dspk.o
ARTS=dspk.so


all: $(ARTS)

dspk.so: $(OBJS) 
	$(CXX) $(LDFLAGS) -o dspk.so $(OBJS) $(LDLIBS) 

dspk.o: $(SRCS)

clean: 
	$(RM) $(OBJS) $(ARTS) 
