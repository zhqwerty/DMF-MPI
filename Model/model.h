#ifndef _MODEL_
#define _MODEL_

#include <armadillo>
using namespace arma;

class Model {
public:
    int taskid;
    int lambda;

    Model(int key, int lamb){
        this->taskid = key;
        this->lambda = lamb;
    };
    virtual ~Model(){};

    virtual mat& getX() = 0;
    virtual mat& getY() = 0;
};

#endif
