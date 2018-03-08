pro test_nospike

path = "/kso/iris_l2_20150615_072426_3610091469_raster_t000_r00000.fits"

FITS_READ, path, data, hdr, EXTEN_NO=4

print, hdr
print, SIZE(data)

ind = [27, 54]

FOREACH i, ind DO BEGIN
  
  nspk = nospike(data[*,*,i])

  atv, nspk

  
  stop
  
ENDFOREACH


;atv, data



end