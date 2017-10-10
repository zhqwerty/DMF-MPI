#ifndef _GRADIENT_
#define _GRADIENT_

#include <armadillo>
using namespace arma;

class Gradient{
public:
    mat data;
    
    Gradient(){};
    virtual ~Gradient(){};
    
    void clear(){
        data.reset();
    }
};

#endif
