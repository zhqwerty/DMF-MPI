#include <iostream>
#include <random>
#include <armadillo>
#include <cmath>
#include "examples.h"
#include "global_macros.h"
#include "model.h"
#include "gradient.h"
using namespace arma;

class Updater {
protected:
    Model* model;
    Example* example;

public:
    Updater(Model* model, Example* example){
        this->model = model;
        this->example = example;
    }

    Updater(){}
    virtual ~Updater(){}

    virtual void UpdateX(Gradient* gradient, int idx, double learning_rate){
        mat& X = model->getX();
        X.row(idx) -= learning_rate * gradient->data;
    }
    
    virtual void UpdateY(Gradient* gradient, int idx, double learning_rate){
        mat& Y = model->getY();
        Y.col(idx) -= learning_rate * gradient->data;
    }


};

void update(Example* trainData, int nExamples, mat& X, mat& Y, double learningRate, double lambda){
    std::random_device rd;
    int randPick = rd() % nExamples;
    mat predict = X.row(trainData[randPick].row) * Y.col(trainData[randPick].col);
    mat gradXi = (predict(0, 0) - trainData[randPick].rating) * Y.col(trainData[randPick].col).t() + lambda * X.row(trainData[randPick].row);
    X.row(trainData[randPick].row) -= learningRate * gradXi;
    mat gradYj = (predict(0, 0) - trainData[randPick].rating) * X.row(trainData[randPick].row).t() + lambda * Y.col(trainData[randPick].col);
    Y.col(trainData[randPick].col) -= learningRate * gradYj;
}

void update_sig(Example* trainData, int nExamples, mat& X, mat& Y, double learningRate, double lambda){
    std::random_device rd;
    int randPick = rd() % nExamples;
    mat predict = X.row(trainData[randPick].row) * Y.col(trainData[randPick].col);
    double den = pow(1 + exp(predict(0, 0) * trainData[randPick].rating), 2);
    mat gradXi = -exp(trainData[randPick].rating * predict(0, 0)) * Y.col(trainData[randPick].col).t() / den + lambda * X.row(trainData[randPick].row); 
    X.row(trainData[randPick].row) -= learningRate * gradXi;
    mat gradYj = -exp(trainData[randPick].rating * predict(0, 0)) * X.row(trainData[randPick].row).t() / den + lambda * Y.col(trainData[randPick].col);
    Y.col(trainData[randPick].col) -= learningRate * gradYj;
}

