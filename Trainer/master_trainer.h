#ifndef _MASTER_TRAINER_
#define _MASTER_TRAINER_

#include "../Example/examples.h"
#include "../Model/model.h"
#include "../Tools/tools.h"
#include "mpi.h"
#include <armadillo>
#include <random>
using namespace arma;

class MasterTrainer : public Trainer {
public:
    Example* testData;
    MasterTrainer(Model* model, Example* trainData, Example* testData) : Trainer(model, trainData){
        testData = testData;
    }

    TrainDtatistics Train(Model* model, Example* example, Updater* updater) override {
        TrainStatistics stats;

        MPI_Status status;
    
    }


}

#endif
