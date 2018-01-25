DSPK_C=img/dspk/dspk.cpp
DSPK_H=img/dspk/dspk.h
DSPK_O=img/dspk/dspk.o


DSPK_SRCS=$(DSPK_C)
DSPK_HDRS=$(DSPK_H)
DSPK_OBJS=$(DSPK_O)
DSPK_LIBS=$(CONVOL_SHRD)
DSPK_LDLIBS=$(addprefix -l:,$(DSPK_LIBS))
DSPK_SHRD=img/dspk/dspk.so

IMG_ARTS+=$(DSPK_SHRD)
IMG_OBJS+=$(DSPK_OBJS)



$(DSPK_SHRD): $(DSPK_OBJS) $(DSPK_LIBS)
	$(CXX) $(LDFLAGS) -o $(DSPK_SHRD) $(DSPK_OBJS) $(LDLIBS) $(DSPK_LDLIBS) 

$(DSPK_O): $(DSPK_C) $(DSPK_H) 
	$(CXX) $(CXXFLAGS) -c $(DSPK_C) -o $(DSPK_O) 
