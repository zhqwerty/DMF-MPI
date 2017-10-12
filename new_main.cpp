#include "defines.h"
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
    
    Model* model = new Model;
    Updater* updater = new Updater;
    Trainer* trainer = NULL;
    if (taskid == 0) {
        trainer = new ServerTrainer(model, trainData, testDta);
    }
    else {
        trainer = new WorkerTrainer(model, trainData);
    }
    TrainStatistics stats = trainer->Train(model, trainData, updater);
    
    MPI_Finalize();
    return 0;
}
