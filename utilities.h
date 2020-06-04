#ifndef GPUSSJOIN_UTILITIES_H
#define GPUSSJOIN_UTILITIES_H

#include <cstddef>

template <typename I>
struct memory_calculator
{
    long double size;
    char scale;

    memory_calculator(double sizeArg, char scaleArg)
            : size(sizeArg), scale(scaleArg) {}

    inline size_t numberOfElements() {
        if (scale == 'M') {
            size *= 1000000.0;
        } else {
            size *= 1000000000.0;
        }
        return static_cast<size_t>(size / (sizeof (I)));
    }
};

typedef memory_calculator<unsigned int> default_memory_calculator;



#endif //PROJECT_UTILITIES_H
