#ifndef _MODEL_
#define _MODEL_

#include <armadillo>
using namespace arma;

class Model {
public:
    int taskid;

    Model(int key){
        this->taskid = key;
    };
    virtual ~Model(){};

    virtual mat& getX() = 0;
    virtual mat& getY() = 0;
};

#endif
