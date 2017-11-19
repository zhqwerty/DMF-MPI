#include <iostream>
#include <vector>
#include <armadillo>
#include <math.h>
#include <mpi.h>
#include "Example/examples.h"
#include "Tools/tools.h"
#include "Updater/updater.h"
#include "Tools/global_macros.h"
using namespace arma;

int main(int argv, char *argc[]){
    int flag, tot;
    MPI_Status state;
    MPI_Init(&argv, &argc);
    MPI_Comm_rank(MPI_COMM_WORLD, &flag);
    MPI_Comm_size(MPI_COMM_WORLD, &tot);
    const char* inputTrainFile = "/Users/ZMY/data/Slashdot/train.txt";
    const char* inputTestFile = "/Users/ZMY/data/Slashdot/test.txt";
    int nRows, nCols, nExamples;
    int testRows, testCols, testExamples;
    Example* trainData = load_examples(inputTrainFile, nRows, nCols, nExamples);
    Example* testData = load_examples(inputTestFile, testRows, testCols, testExamples);
    
    std::cout << "nRows: " << nRows  << " nCols: " << nCols << " nExamples: " << nExamples << std::endl;
    std::cout << "testRows: " << testRows  << " testCols: " << testCols << " testExamples: " << testExamples << std::endl;
//    for (int i = 0; i < 10; i++) std::cout << trainData[i].row << " " << trainData[i].col << " " << trainData[i].rating << std::endl;
    
    int rank = 20;
    int edges = 10000;
    int batchNum = 3;
    int nodes = 4;
    double lambda = 0.1;
    double learningRate = 1; 
    mat X = randn<mat>(nRows, rank);
    mat Y = randn<mat>(rank, nCols);
    
    // Variables Update
    bool convergence = false;
    int iter = 1;
    int maxIter = 2e6;    
    int numMaker = 20;
    std::vector<double> acc;
    std::vector<double> rmse;
     
    std::cout << "Start Training ... " << std::endl;
    Timer trainTime;
    trainTime.Tick();
    while (!convergence){
        learningRate = 10 / pow(iter, 0.1);
        // update ...
        update_sig(trainData, nExamples, X, Y, learningRate, lambda);
        if (iter > maxIter) convergence = true;
        if (iter % (maxIter / numMaker) == 0){
            int trueNum = 0;
            long double error = 0;
            for (int i = 0; i < testExamples; i++){
                mat predict = X.row(testData[i].row) * Y.col(testData[i].col);
//                rowvec X_row = X.row(testData[i].row);
//                colvec Y_col = Y.col(testData[i].col);
//                X_row.print("X_row: \n");
//                Y_col.print("Y_col: \n");
//                predict.print("predict: \n");
                DEBUG_ONLY(std::cout << "predict value: " << predict(0, 0) << std::endl;);
                if ( predict(0, 0) * testData[i].rating > 0 ) trueNum++;
                error += pow(predict(0, 0) - testData[i].rating, 2);
                //std::cout << "error: " << error << std::endl;
                DEBUG_ONLY(std::cout << "error = " << predict(0, 0) << " - " << testData[i].rating << std::endl;);
            }
//            std::cout << "trueNum: " << trueNum << std::endl;
            acc.push_back(double(trueNum) / testExamples);
            rmse.push_back(sqrt(error) / testExamples);
            std::cout << "Iter: " << iter << "  Accuracy: " << acc.back() << "    RMSE: " << rmse.back() << std::endl;
        }
        iter++;
    }
    trainTime.Tock();
    std::cout << "Training time spend: " << trainTime.duration << " s" << std::endl;
    DEBUG_ONLY(printVec(acc););
    DEBUG_ONLY(printVec(rmse););
    MPI_Finalize();
    return 0;
}
