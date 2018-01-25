DSPK_DIR=img/dspk/
DSPK_ROOT=/../../

DSPK_BASE=dspk
DSPK=$(addprefix $(DSPK_DIR),$(DSPK_BASE))
DSPK_C=$(addprefix $(DSPK),.cpp)
DSPK_H=$(addprefix $(DSPK),.h)
DSPK_O=$(addprefix $(DSPK),.o)

DSPK_SRCS=$(DSPK_C)
DSPK_HDRS=$(DSPK_H)
DSPK_OBJS=$(DSPK_O)
DSPK_SHRD=$(addsuffix .so,$(DSPK))

DSPK_LIBS=$(CONVOL_SHRD) $(UTIL_SHRD)
DSPK_LDIRS=$(dir $(DSPK_LIBS))
DSPK_LFILES=$(notdir $(DSPK_LIBS))
DSPK_RP_ROOT=$(addprefix $(RP_ORIG), $(DSPK_ROOT))
DSPK_RPATH=$(addprefix $(DSPK_RP_ROOT), $(DSPK_LDIRS))
DSPK_Wl = $(addprefix -Wl$(COMMA)-rpath$(COMMA),$(addprefix ',$(addsuffix ',$(DSPK_RPATH))))
DSPK_LDLIBS=$(addprefix -L./,$(DSPK_LDIRS)) $(addprefix -l:,$(DSPK_LFILES)) $(DSPK_Wl) 



IMG_ARTS+=$(DSPK_SHRD)
IMG_OBJS+=$(DSPK_OBJS)



$(DSPK_SHRD): $(DSPK_OBJS) $(DSPK_LIBS) 
	@echo $(UTIL_SHRD) 
	$(CXX) $(LDFLAGS) -o $(DSPK_SHRD) $(DSPK_OBJS)  $(LDLIBS) $(DSPK_LDLIBS)

$(DSPK_O): $(DSPK_C) $(DSPK_H) 
	$(CXX) $(CXXFLAGS) -c $(DSPK_C) -o $(DSPK_O) 
