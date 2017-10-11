#ifndef _WORKER_TRAINER_
#define _WORKER_TRAINER_

#include "gradient.h"
#include "examples.h"
#include <armadillo>
#include <random>
#include "tools.h"
using namespace arma;

class Message {
public:
    mat Xi;
    mat Yj;
    Message(mat xi, mat yj){
        this->Xi = xi;
        this->Yj = yj;
    }
    ~Message(){}
}

class WorkerTrainer : public Trainer {
public:
    WorkerTrainer(Model* model, Example* example) : Trainer(model, example){}
    ~WorkerTrainer(){}
    
    TrainStatistics Train(Model* model, Example* example, Updater* updater) override {
        TrainStatistics stats;

        MPI_Status status;
        Message message;

        std::random_device rd;
        int pick = rd() % model.nExamples();
        MPI_Recv(&message[]);
        double learning_rate = 1e-1;
        updater->Update(message.Xi, message.Yj, &example[pick], learning_rate, model.lambda);
        
        MPI_Send(&message, size, MPI_DOUBLE, MPI_COMM_WORLD);
        
        return stats;

    } 

}

#endif
