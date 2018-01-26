COMMA = ,

CXX=g++
RM=rm -f
CXXFLAGS=-O3 -I/usr/include/python3.5m/ -I. -fPIC
LDFLAGS=-O3 -shared
LDLIBS=-lboost_python-py35 -lboost_numpy3 -L.

RP_ORIG=$$ORIGIN

KSO_MK_VARS = img/img_vars.mk
KSO_MK_VARS += util/util_vars.mk

KSO_MK_RULES = img/img_rules.mk
KSO_MK_RULES += util/util_rules.mk

include $(KSO_MK_VARS)
include $(KSO_MK_RULES) 



all: $(KSO_ARTS)

clean: 
	$(RM) $(KSO_OBJS) $(KSO_ARTS) 
