$(UTIL_SHRD): $(UTIL_OBJS)
	$(CXX) $(LDFLAGS) -o $(UTIL_SHRD) $(UTIL_OBJS) $(LDLIBS)

$(DIM3_O): $(DIM3_C) $(DIM3_H) 
	$(CXX) $(CXXFLAGS) -c $(DIM3_C) -o $(DIM3_O) 
