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

    Model(double lambda, int nRows, int nCols, int nExamples, int rank) {
        lambda = lambda;
        rank = rank;
        nRows = nRows;
        nCols = nCols;
        nExamples = nExamples;
        X = randn<mat>(nRows, rank);
        Y = randn<mat>(rank, nCols);
    }
};

#endif
