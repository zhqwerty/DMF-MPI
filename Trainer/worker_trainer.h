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
        double flag_epoch = 0;
        double flag_break = 0;
        MPI_Status status;
        // message: Xi, Yj, idx, learning_rate, flag_epoch, flag_break;
        std::vector<double> message(model->rank * 2 + 4);
        
        while (true) { 
            if (flag_epoch){
                flag_epoch = 0;
                continue;
            }
            if (flag_break){
                break;
            }
            // std::cout << "worker receive from master" << std::endl;
            MPI_Recv(&message[0], model->rank * 2 + 4, MPI_DOUBLE, 0, 102, MPI_COMM_WORLD, &status);
            // std::cout << "worker receive down" << std::endl;
          
            // Prase message
            std::vector<double> tmp_xi(message.begin(), message.begin() + model->rank);
            std::vector<double> tmp_yj(message.begin() + model->rank, message.begin() + 2 * model->rank);
            
            mat Xi = vec_2_mat(tmp_xi, 0, 1, model->rank);
            mat Yj = vec_2_mat(tmp_yj, 0, model->rank, 1);
            
            int idx = *(message.end() - 4);
            double learning_rate = *(message.end() - 3);
            flag_epoch = *(message.end() - 2);
            flag_break = *(message.end() - 1);

            // Update Xi and Yj
            updater->Update_Sig(Xi, Yj, &trainData[idx], learning_rate, model->lambda);

            tmp_xi = mat_2_vec(Xi);
            tmp_yj = mat_2_vec(Yj);
            
            // Send (Xi, Yj, idx)
            tmp_xi.insert(tmp_xi.end(), tmp_yj.begin(), tmp_yj.end());
            tmp_xi.push_back(idx);
            // std::cout << "worker send to master" << std::endl;
            MPI_Send(&tmp_xi[0], model->rank * 2 + 1, MPI_DOUBLE, 0, 101, MPI_COMM_WORLD);
            // std::cout << "worker send down" << std::endl;

        }
        return stats;
    } 
};

#endif
