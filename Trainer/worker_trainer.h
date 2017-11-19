#ifndef _WORKER_TRAINER_
#define _WORKER_TRAINER_

#include "../Example/examples.h"
#include "../Model/model.h"
#include "../Tools/tools.h"
#include "mpi.h"
#include <armadillo>
using namespace arma;

class WorkerTrainer : public Trainer {
public:
    WorkerTrainer(Model* model, Example* trainData) : Trainer(model, trainData){}
    ~WorkerTrainer(){}
    
    TrainStatistics Train(Model* model, Example* trainData, Updater* updater) override {
        TrainStatistics stats;
        //std::cout << "worker_trainer running... " <<  std::endl;
        double flag_break = 0;
        MPI_Status status;
        // message: Xi, Yj, idx, flag_break;
        std::vector<double> message(model->rank * 2 + 2);
        
        while (true) { 
            if (flag_break) break;
            // std::cout << "worker receive from master" << std::endl;
            MPI_Recv(&message[0], model->rank * 2 + 2, MPI_DOUBLE, 0, 102, MPI_COMM_WORLD, &status);
            // std::cout << "worker receive down" << std::endl;
          
            // Prase message
            std::vector<double> tmp_xi(message.begin(), message.begin() + model->rank);
            std::vector<double> tmp_yj(message.begin() + model->rank, message.begin() + 2 * model->rank);
            
            mat Xi = vec_2_mat(tmp_xi, 0, 1, model->rank);
            mat Yj = vec_2_mat(tmp_yj, 0, model->rank, 1);
            
            int idx = *(message.end() - 2);
            flag_break = *(message.end() - 1);

            // Calculate Gradient
            std::pair<mat, mat> gradient;
            gradient = updater->CalGradient(Xi, Yj);

            std::vector<double> gradXi = mat_2_vec(gradient.first);
            std::vector<double> gradYj = mat_2_vec(gradient.second);
            
            // Send (gradXi, gradYj, idx)
            gradXi.insert(gradXi.end(), gradYj.begin(), gradYj.end());
            gradXi.push_back(idx);
            // std::cout << "worker send to master" << std::endl;
            MPI_Send(&gradXi[0], model->rank * 2 + 1, MPI_DOUBLE, 0, 101, MPI_COMM_WORLD);
            // std::cout << "worker send down" << std::endl;
        }
        return stats;
    } 
};

#endif
