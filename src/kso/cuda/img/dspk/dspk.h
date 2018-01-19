/*
 * dspk.h
 *
 *  Created on: Jan 19, 2018
 *      Author: byrdie
 */

#ifndef DSPK_H_
#define DSPK_H_

#include <boost/python.hpp>

namespace kso {

	namespace img {

		namespace dspk {

			void remove_noise();

			BOOST_PYTHON_MODULE(dspk){

			    boost::python::def("remove_noise", remove_noise);

			}

		}

	}

}



#endif /* DSPK_H_ */
