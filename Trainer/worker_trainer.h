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
    WorkerTrainer(Model* model, Example* example) : Trainer(model, example){}
    ~WorkerTrainer(){}
    
    TrainStatistics Train(Model* model, Example* example, Updater* updater) override {
        TrainStatistics stats;
        std::cout << "worker_trainer running... " <<  std::endl;
        double flag_epoch = 1;
        double flag_break = 0;
        std::cout << "worker 111" << std::endl;
        MPI_Status status;
        // message: Xi, Yj, idx, learning_rate, flag_epoch, flag_break;
        std::vector<double> message(model->rank * 2 + 4);
        
        std::cout << "worker 222" << std::endl;
        while (true) { 
            if (flag_epoch){
                flag_epoch = 0;
                continue;
            }
            if (flag_break){
                break;
            }
            
            std::cout << "worker 333" << std::endl;
            MPI_Recv(&message[0], model->rank + 4, MPI_DOUBLE, 0, 102, MPI_COMM_WORLD, &status);
            // Prase message
            std::vector<double> tmp_xi(message.begin(), message.begin() + model->rank);
            std::vector<double> tmp_yj(message.begin() + model->rank, message.begin() + 2 * model->rank);
            mat Xi = vec_2_mat(tmp_xi, 0, 1, model->rank);
            mat Yj = vec_2_mat(tmp_yj, 0, model->rank, 1);

            int idx = *(message.end() - 4);
            double learning_rate = *(message.end() - 3);
            flag_epoch = *(message.end() - 2);
            flag_break = *(message.end() - 1);

            std::cout << "worker 444" << std::endl;
            // Update Xi and Yj
            updater->Update(Xi, Yj, &example[idx], learning_rate, model->lambda);

            tmp_xi = mat_2_vec(Xi);
            tmp_yj = mat_2_vec(Yj);
            std::cout << "Xi :";
            printVec(tmp_xi);
            std::cout << std::endl;

            tmp_xi.insert(tmp_xi.end(), tmp_yj.begin(), tmp_yj.end());
            tmp_xi.push_back(idx);
            MPI_Send(&tmp_xi[0], tmp_xi.size(), MPI_DOUBLE, 0, 101, MPI_COMM_WORLD);

        }
        return stats;
    } 
};

#endif
