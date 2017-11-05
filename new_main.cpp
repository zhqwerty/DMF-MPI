#include <iostream>
#include <vector>
#include <armadillo>
#include <math.h>
#include <mpi.h>
#include "Example/examples.h"
#include "Updater/updater.h"
#include "Trainer/trainer.h"
#include "Trainer/worker_trainer.h"
#include "Trainer/master_trainer.h"
#include "Tools/global_macros.h"

using namespace arma;

int main(int argv, char *argc[]){
    int taskid, numtasks;
    MPI_Status state;
    MPI_Init(&argv, &argc);
    MPI_Comm_rank(MPI_COMM_WORLD, &taskid);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
 
    const char* inputTrainFile = "/Users/ZMY/Downloads/Matrix/data/Slashdot/train.txt";
    const char* inputTestFile = "/Users/ZMY/Downloads/Matrix/data/Slashdot/test.txt";
    int nRows, nCols, nExamples;
    int testRows, testCols, testExamples;
    Example* trainData = load_examples(inputTrainFile, nRows, nCols, nExamples);
    Example* testData = load_examples(inputTestFile, testRows, testCols, testExamples);
    
    std::cout << "nRows: " << nRows  << " nCols: " << nCols << " nExamples: " << nExamples << std::endl;
    std::cout << "testRows: " << testRows  << " testCols: " << testCols << " testExamples: " << testExamples << std::endl;
//    for (int i = 0; i < 10; i++) std::cout << trainData[i].row << " " << trainData[i].col << " " << trainData[i].rating << std::endl;
     
   
    double lambda = 1e-1;
    int rank = 20;

    Model* model = new Model(lambda, nRows, nCols, nExamples, rank);
    Updater* updater = new Updater(model, trainData);
    Trainer* trainer = NULL;
    if (taskid == 0) {
        trainer = new MasterTrainer(model, trainData, testData, testExamples);
    }
    else {
        trainer = new WorkerTrainer(model, trainData);
    }
    TrainStatistics stats = trainer->Train(model, trainData, updater);
   
    // Delete new data
    delete model;
    delete updater;
    delete trainer;
    delete trainData;
    delete testData;
    
    MPI_Finalize();
    return 0;
}
