
#include <stdio.h>
#include <stdlib.h>


#include <boost/python.hpp>

char const* greet()
{
   return "hello, world";
}

BOOST_PYTHON_MODULE(test_pyboost)
{
    using namespace boost::python;

    def("greet", greet);
}

