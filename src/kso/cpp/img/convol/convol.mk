CONVOL_DIR=img/convol/
CONVOL_ROOT=/../../

CONVOL_BASE=convol
CONVOL=$(addprefix $(CONVOL_DIR),$(CONVOL_BASE))
CONVOL_C=$(addsuffix .cpp,$(CONVOL))
CONVOL_H=$(addsuffix .h,$(CONVOL))
CONVOL_O=$(addsuffix .o,$(CONVOL))


CONVOL_SRCS=$(CONVOL_C)
CONVOL_HDRS=$(CONVOL_H)
CONVOL_OBJS=$(CONVOL_O)
CONVOL_SHRD=$(addsuffix .so,$(CONVOL))

IMG_ARTS+=$(CONVOL_SHRD)
IMG_OBJS+=$(CONVOL_OBJS)



$(CONVOL_SHRD): $(CONVOL_OBJS)
	$(CXX) $(LDFLAGS) -o $(CONVOL_SHRD) $(CONVOL_OBJS) $(LDLIBS)

$(CONVOL_O): $(CONVOL_C) $(CONVOL_H) 
	$(CXX) $(CXXFLAGS) -c $(CONVOL_C) -o $(CONVOL_O) 
