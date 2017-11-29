#include <iostream>
#include <vector>
#include <armadillo>
#include <math.h>
#include <random>
#include <mpi.h>
#include "Example/examples.h"
#include "Tools/tools.h"
#include "Updater/updater.h"
#include "Tools/global_macros.h"
using namespace arma;

int main(int argv, char *argc[]){
    const char* inputFile = "/home/han/data/Slashdot/slashdot.txt";
    int nRows, nCols, nExamples;
    Example* examples = load_examples(inputFile, nRows, nCols, nExamples);
    
    std::cout << "nRows: " << nRows  << " nCols: " << nCols << " nExamples: " << nExamples << std::endl;
 //   std::cout << "testRows: " << testRows  << " testCols: " << testCols << " testExamples: " << testExamples << std::endl;
//    for (int i = 0; i < nExamples; i++) std::cout << examples[i].row << " " << examples[i].col << " " << examples[i].rating << std::endl;
    
    int rank = 20;
    double sample_rate = 0.9;
    double lambda = 0.1;
    mat X = randn<mat>(nRows, rank);
    mat Y = randn<mat>(rank, nCols);
   
    std::vector<int> sample(nExamples, 0);
    for (int i = 0; i < nExamples; i++) sample[i] = i;
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(sample.begin(), sample.end(), g);

    // Variables Update
    int maxIter = 1e5;   
    int maxEpoch = 40;
    double learning_rate = 1;
    std::vector<double> acc;
    std::vector<double> rmse;
     
    std::cout << "Start Training ... " << std::endl;
    Timer trainTime;
    trainTime.Tick();
    for (int epoch = 1; epoch <= maxEpoch; epoch++){
        for (int iter = 1; iter <= maxIter; iter++){
            learning_rate = 10 / pow(maxIter * (epoch - 1) + iter, 0.1);
            // update ...
            update_sig(examples, sample, sample_rate, nExamples, X, Y, learning_rate, lambda);
        }
        // Test Error and Accuracy 
        int trueNum = 0;
        long double error = 0;
        for (int i = int(nExamples * sample_rate) + 1; i < nExamples; i++){
            int pick = sample[i]; 
            mat predict = X.row(examples[pick].row) * Y.col(examples[pick].col);
            if ( sign(predict(0, 0)) == sign(examples[pick].rating) ) trueNum++;
            error += pow(predict(0, 0) - examples[pick].rating, 2);
        }
        int testExamples = nExamples * (1 - sample_rate);
        acc.push_back(double(trueNum) / testExamples);
        rmse.push_back(sqrt(error / testExamples));
        trainTime.Tock();
        printf("Epoch: %d   Accuracy: %.4f  RMSE: %.4f  Spend Time: %.2f s \n", epoch, acc.back(), rmse.back(), trainTime.duration); 
    }
    return 0;
}
