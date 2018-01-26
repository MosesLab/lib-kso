UTIL_DIR=util/
UTIL_ROOT=/../
UTIL_BASE=util
UTIL=$(addprefix $(UTIL_DIR),$(UTIL_BASE))

DIM3_BASE=dim3
DIM3=$(addprefix $(UTIL_DIR),$(DIM3_BASE))
DIM3_C=$(addsuffix .cpp,$(DIM3))
DIM3_H=$(addsuffix .h,$(DIM3))
DIM3_O=$(addsuffix .o,$(DIM3))


UTIL_SRCS=$(DIM3_C)
UTIL_HDRS=$(DIM3_H)
UTIL_OBJS=$(DIM3_O)
UTIL_SHRD=$(addsuffix .so,$(UTIL))

KSO_ARTS+=$(UTIL_SHRD)
KSO_OBJS+=$(UTIL_OBJS)
