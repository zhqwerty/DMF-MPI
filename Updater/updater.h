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
    
    virtual void ApplyGradient(Model& model, const int row, const int col, const mat& gradXi, const mat& gradYj, const double learning_rate){
        model.X.row(row) -= learning_rate * gradXi;
        model.Y.col(col) -= learning_rate * gradYj;
    }

    std::pair<mat, mat> CalGradient(mat& Xi, mat& Yj, const int idx){
        std::pair<mat, mat> res;
        mat predict = Xi * Yj;
        double den = pow(1 + exp(predict(0, 0) * example[idx].rating), 2);
        mat gradXi = -exp(example[idx].rating * predict(0, 0)) * example[idx].rating *  Yj.t() / den + model->lambda * Xi;
        mat gradYj = -exp(example[idx].rating * predict(0, 0)) * example[idx].rating *  Xi.t() / den + model->lambda * Yj;
//        std::cout << "example->rating: " << example->rating << std::endl; 
        res.first = gradXi;
        res.second = gradYj;
        return res;
    }

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
        mat gradXi = -exp(example->rating * predict(0, 0)) * example->rating * Yj.t() / den + lambda * Xi;
        Xi -= learning_rate * gradXi;
        predict = Xi * Yj;
        mat gradYj = -exp(example->rating * predict(0, 0)) * example->rating * Xi.t() / den + lambda * Yj;
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

void update_sig(Example* trainData, const std::vector<int>& sample, double sample_rate, int nExamples, mat& X, mat& Y, double learningRate, double lambda){
    std::random_device rd;
    int randPick = rd() % (int(nExamples * sample_rate));
    int i = trainData[sample[randPick]].row, j = trainData[sample[randPick]].col, sign = trainData[sample[randPick]].rating;
    mat predict = X.row(i) * Y.col(j);
    double den = pow(1 + exp(predict(0, 0) * sign), 2);
    mat gradXi = -exp(sign * predict(0, 0)) * sign * Y.col(j).t() / den + lambda * X.row(i); 
    X.row(i) -= learningRate * gradXi;
    mat gradYj = -exp(sign * predict(0, 0)) * sign * X.row(i).t() / den + lambda * Y.col(j);
    Y.col(j) -= learningRate * gradYj;
}

