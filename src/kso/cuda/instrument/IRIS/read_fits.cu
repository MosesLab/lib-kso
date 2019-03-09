
#include "read_fits.h"

namespace kso {

namespace instrument {

namespace IRIS {


using namespace std;
//using namespace CCfits;



dim3 read_fits_raster(string path, float * buf){

	int bitpix;

	fitsfile *fptr;
	char card[FLEN_CARD];
	int status = 0,  nkeys, ii;  /* MUST initialize status */
	int num_hdus, i_h, type_h;

	fits_open_file(&fptr, path.c_str(), READONLY, &status);

	fits_get_num_hdus(fptr, &num_hdus, & status);
	cout << "Number of HDUs: " << num_hdus << endl;

	// find number of windows
	uint nwin;
	fits_read_key(fptr, TUINT, "NWIN", &nwin, NULL, &status);
	cout << "Number of windows " << nwin << endl;

	const char * window = "'Si IV 1403'";
	char keyval[256];
	char key[256];

	// loop through all windows
	uint i;
	for(i = 1; i <= nwin; i++){

		sprintf(key, "TDESC%d", i);
		fits_read_keyword(fptr, key, keyval, NULL, &status);

//		printf("%s\n", keyval);

		if(strcmp(keyval, window) == 0){
			i = i + 1;		// add one since the first HDU has no data
			break;
		}

	}



//	cout << "_______________________" << endl;

	fits_movabs_hdu(fptr, i, &type_h, &status);

	fits_get_hdu_num(fptr, &i_h);
//	cout << "Current HDU index: " << i_h << endl;

	//	fits_get_hdu_type(fptr, &type_h, &status);
//	cout << "Current HDU type: " << type_h << endl;

	fits_get_hdrspace(fptr, &nkeys, NULL, &status);
//	cout << "Number of keys: " << nkeys << endl;


	fits_get_img_type(fptr, &bitpix, &status);
//	cout << "Image type: " << bitpix << endl;

	for (ii = 1; ii <= nkeys; ii++)  {
		fits_read_record(fptr, ii, card, &status); /* read keyword */
//		printf("%s\n", card);
	}
//	printf("END\n\n");  /* terminate listing with END */

	dim3 sz;
	long axis_sz[3];
	fits_get_img_size(fptr, 3, axis_sz, &status);
	sz.x = axis_sz[0];
	sz.y = axis_sz[1];
	sz.z = axis_sz[2];
	long sz3 = sz.x * sz.y * sz.z;



	long fpixel[3];
	fpixel[0] = 1;
	fpixel[1] = 1;
	fpixel[2] = 1;
	fits_read_pix(fptr, TFLOAT, fpixel, sz3, NULL, buf, NULL, &status);



	fits_close_file(fptr, &status);

	if (status)          /* print any error messages */
		fits_report_error(stderr, status);
	//	return(status);

	return sz;

}

void read_fits_raster_ndarr(np::ndarray & nd_buf, np::ndarray & nd_sz){

	string path = "/kso/iris_l2_20150615_072426_3610091469_raster_t000_r00000.fits";
	//	string path = "/kso/iris_l2_20140125_030458_3860259280_raster_t000_r00000.fits";
	//	string path = "/kso/iris_l2_20140404_001944_3800259353_raster_t000_r00000.fits";

	float * buf = (float *) nd_buf.get_data();

	dim3 sz = read_fits_raster(path, buf);

	nd_sz[0] = (uint) sz.z;
	nd_sz[1] = (uint) sz.y;
	nd_sz[2] = (uint) sz.x;

}


}

}

}
