pro test_dspk

path = "/kso/iris_l2_20150615_072426_3610091469_raster_t000_r00000.fits"

FITS_READ, path, data, hdr, EXTEN_NO=4

print, hdr
print, SIZE(data)



data = dspk(data, sigmas=3.0, Niter=5, mode='both')


atv, data[*,*,54]

xstepper, data, xsize=1920 * 0.6, ysize = 1080 * 0.6

end