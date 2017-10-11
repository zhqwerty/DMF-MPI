#ifndef _MODELMFSIG_
#define _MODELMFSIG_

#include <armadillo>
using namespace arma;

class Model_MF_Sig : public Model {
public:
    mat X;
    mat Y;
    int rank;
    int nRows;
    int nCols;
    int nExamples;

    Model_MF_Sig(int taskid, int nRows, int nCols, int nExamples, int rank) : Model(taskid){
        rank = rank;
        nRows = nRows;
        nCols = nCols;
        nExamples = nExamples;
        X = randn<mat>(nRows, rank);
        Y = randn<mat>(rank, nCols);
    }
}

#endif
