#ifndef _MODELMFSIG_
#define _MODELMFSIG_

#include <armadillo>
using namespace arma;

class Model_MF_Sig : public Model {
private:
    mat _X;
    mat _Y;
    int _rank;
    int _nRows;
    int _nCols;
    
public:
    Model_MF_Sig(int taskid, int nRows, int nCols, int rank) : Model(taskid){
        _rank = rank;
        _nRows = nRows;
        _nCols = nCols;
        _X = randn<mat>(nRows, rank);
        _Y = randn<mat>(rank, nCols);
    }
    
    mat getX() override{
        return _X;
    }
    mat getY() override{
        return _Y;
    }
}


#endif
