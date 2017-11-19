#include <iostream>
#include <random>
#include <armadillo>
#include <cmath>
#include "../Example/examples.h"
#include "../Tools/global_macros.h"
#include "../Model/model.h"
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

    virtual void Update(mat& Xi, mat& Yj, Example* example, double learning_rate, double lambda){
        mat predict = Xi * Yj;
        mat gradXi = (predict(0, 0) - example->rating) * Yj.t() + lambda * Xi;
        Xi -= learning_rate * gradXi;
        mat gradYj = (predict(0, 0) - example->rating) * Xi.t() + lambda * Yj;
        Yj -= learning_rate * gradYj;
    }
    
    virtual void Update_Sig(mat& Xi, mat& Yj, Example* example, double learning_rate, double lambda){
        mat predict = Xi * Yj;
        double den = pow(1 + exp(predict(0, 0) * example->rating), 2);
        mat gradXi = -exp(example->rating * predict(0, 0)) * Yj.t() / den + lambda * Xi;
        Xi -= learning_rate * gradXi;
        predict = Xi * Yj;
        mat gradYj = -exp(example->rating * predict(0, 0)) * Xi.t() / den + lambda * Yj;
        Yj -= learning_rate * gradYj;
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
    mat gradXi = -exp(trainData[randPick].rating * predict(0, 0)) * trainData[randPick].rating * Y.col(trainData[randPick].col).t() / den + lambda * X.row(trainData[randPick].row); 
    X.row(trainData[randPick].row) -= learningRate * gradXi;
    mat gradYj = -exp(trainData[randPick].rating * predict(0, 0)) * trainData[randPick].rating * X.row(trainData[randPick].row).t() / den + lambda * Y.col(trainData[randPick].col);
    Y.col(trainData[randPick].col) -= learningRate * gradYj;
}

