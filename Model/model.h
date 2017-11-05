#ifndef _MODELMFSIG_
#define _MODELMFSIG_

#include <armadillo>
using namespace arma;

class Model {
public:
    mat X;
    mat Y;
    int rank;
    int nRows;
    int nCols;
    int nExamples;
    double lambda;

    Model(double lambda1, int nRows1, int nCols1, int nExamples1, int rank1) {
        lambda = lambda1;
        rank = rank1;
        nRows = nRows1;
        nCols = nCols1;
        nExamples = nExamples1;
        X = randn<mat>(nRows1, rank1);
        Y = randn<mat>(rank1, nCols1);
    }
};

#endif
