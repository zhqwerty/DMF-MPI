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
    int FLAGS_n_epochs = 2e5;
    int FLAGS_in_iters = 10;
    int FLAGS_num_workers = 3;
    Example* testData;
    MasterTrainer(Model* model, Example* trainData, Example* testData) : Trainer(model, trainData){
        testData = testData;
    }

    TrainStatistics Train(Model* model, Example* trainData, Updater* updater) override {
        TrainStatistics stats;
        MPI_Status status;
        std::cout << "master_trainer running..." << std::endl;    

        Model& master_model = *model;
        // Train.
        std::cout << "master 111" << std::endl;
        for (int epoch = 0; epoch < 2e5; epoch++){
            srand(epoch);
            double learning_rate = 10 / std::pow(1+epoch, 0.1);
            std::vector<int> delay_counter(FLAGS_num_workers, 1);
            std::cout << "Epoch : " << epoch << std::endl;

            std::cout << "master epoch : "  << epoch << std::endl;
            for(int iter_counter = 0; iter_counter < FLAGS_in_iters; iter_counter++){
                
                std::vector<int> cur_received_workers(FLAGS_num_workers, 0);                
               
                std::cout << "master 111" << std::endl;
                // build message : Xi, Yj, idx, learning_rate, flag
                for (int i = 0; i < FLAGS_num_workers; i++){
                    std::vector<double> message;
                    int idx = rand() % model->nExamples;
                    int row = trainData[idx].row;
                    int col = trainData[idx].col;
                    mat Xi = master_model.X.row(row);
                    mat Yj = master_model.Y.col(col);
                    std::vector<double> tmp_xi = mat_2_vec(Xi);
                    std::vector<double> tmp_yj = mat_2_vec(Yj);
                    message.assign(tmp_xi.begin(), tmp_xi.end()); // Xi
                    message.insert(message.end(), tmp_yj.begin(), tmp_yj.end()); // Yj
                    message.push_back(idx); // idx;

                    if (iter_counter < FLAGS_in_iters - 1) {
                        message.push_back(0);
                        message.push_back(0);
                    }
                    else if(epoch < FLAGS_n_epochs - 1) {
                        message.push_back(1);
                        message.push_back(0);
                    }
                    else {
                        message.push_back(1);
                        message.push_back(1);
                    }
                    std::cout << "master 222" << std::endl;
                    // send messages to workers
                    if(cur_received_workers[i] == 0)
                        delay_counter[i] += 1;
                    else {
                        MPI_Send(&message[0], message.size(), MPI_DOUBLE, i+1, 102, MPI_COMM_WORLD);
                    }
                }
                
                std::cout << "master 333" << std::endl;
                // Receive information(Xi, Yj, idx) and update(assign)
                int cur_worker_size = 0;
                std::vector<double> updated_Xi_Yj(model->rank * 2, 0);
                bool flag_receive = true;
                while (flag_receive){
                    MPI_Probe(MPI_ANY_SOURCE, 101, MPI_COMM_WORLD, &status);    
                    int taskid = status.MPI_SOURCE;
                    MPI_Recv(&updated_Xi_Yj[0], updated_Xi_Yj.size(), MPI_DOUBLE, taskid, 101, MPI_COMM_WORLD, &status);
                    
                    // convert vector to mat
                    std::vector<double> tmp_xi(updated_Xi_Yj.begin(), updated_Xi_Yj.begin() + model->rank);
                    std::vector<double> tmp_yj(updated_Xi_Yj.begin() + model->rank, updated_Xi_Yj.begin() + model->rank * 2);
                    mat Xi = vec_2_mat(tmp_xi, 0, 1, model->rank);
                    mat Yj = vec_2_mat(tmp_yj, 0, model->rank, 1);
                    int idx = *(updated_Xi_Yj.end() - 1);
                    
                    // update Xi, Yj in model
                    int row = trainData[idx].row;
                    int col = trainData[idx].col;
                    master_model.X.row(row) = Xi;
                    master_model.Y.col(col) = Yj.t();
                    
                    cur_worker_size += 1;
                    delay_counter[taskid - 1] = 1;
                    cur_received_workers[taskid - 1] = 1;

                    flag_receive = false;

                }
                std::cout << "master 444" << std::endl;
            }
            

        }
        return stats;
    }

};

#endif
