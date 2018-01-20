/*
 * dspk.h
 *
 *  Created on: Jan 19, 2018
 *      Author: byrdie
 */

#ifndef DSPK_H_
#define DSPK_H_

#include <stdio.h>
#include <stdlib.h>

#include <boost/python.hpp>
#include <boost/python/numpy.hpp>


namespace p = boost::python;
namespace np = boost::python::numpy;

namespace kso {

	namespace img {

		namespace dspk {

			void remove_noise();

		}

	}

}



#endif /* DSPK_H_ */
