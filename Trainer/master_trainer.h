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
    int FLAGS_n_epochs = 2e1;
    int FLAGS_in_iters = 1;
    int FLAGS_num_workers = 3;
    int FLAGS_group_size = 1;
    int FLAGS_max_delay = 2;
    int numMaker = 20;
    int testExamples;
    Example* testData;
    MasterTrainer(Model* model, Example* trainData, Example* testData1, int testExamples1) : Trainer(model, trainData){
        testData = testData1;
        testExamples = testExamples1;
    }

    TrainStatistics Train(Model* model, Example* trainData, Updater* updater) override {
        TrainStatistics stats;
        MPI_Status status;
        //std::cout << "master_trainer running..." << std::endl;    

        Model& master_model = *model;
        // Train.
        for (int epoch = 0; epoch <= FLAGS_n_epochs; epoch++){
            double learning_rate = 10 / std::pow(1+epoch, 0.1);
            //std::cout << "Epoch: " << epoch << std::endl;
            //std::cout << "master learning-rate: " << learning_rate << std::endl;
            std::vector<int> delay_counter(FLAGS_num_workers, 1);

            for(int iter_counter = 0; iter_counter < FLAGS_in_iters; iter_counter++){
                int cur_worker_size = 0;
                std::vector<int> cur_received_workers(FLAGS_num_workers, 0);                
               
                // build message : Xi, Yj, idx, learning_rate, flag_epoch, flag_break
                for (int i = 1; i <= FLAGS_num_workers; i++){
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
                    message.push_back(learning_rate); // learning_rate

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
                    // send messages to workers
                    //printf("master send message to worker %d \n", i + 1);
                    MPI_Send(&message[0], model->rank * 2 + 4, MPI_DOUBLE, i, 102, MPI_COMM_WORLD);
                    //printf("master send down\n");
                }
                
                // Receive info(Xi, Yj, idx) and update(assign)
                std::vector<double> info(model->rank * 2 + 1, 0);
                bool flag_receive = true;
                while (flag_receive){
                    MPI_Probe(MPI_ANY_SOURCE, 101, MPI_COMM_WORLD, &status);    
                    int taskid = status.MPI_SOURCE;
                    //for (int i = 0; i < FLAGS_num_workers; i++){
                        //printf("master receive from worker %d \n", taskid);
                        MPI_Recv(&info[0], model->rank * 2 + 1, MPI_DOUBLE, taskid, 101, MPI_COMM_WORLD, &status);
                        // Prase updaterd_Xi_Yj(Xi, Yj, idx);
                        // convert vector to mat
                        std::vector<double> tmp_xi(info.begin(), info.begin() + model->rank);
                        std::vector<double> tmp_yj(info.begin() + model->rank, info.begin() + model->rank * 2);
                        mat Xi = vec_2_mat(tmp_xi, 0, 1, model->rank);
                        mat Yj = vec_2_mat(tmp_yj, 0, model->rank, 1);
                        int idx = *(info.end() - 1);
                    
                        // update Xi, Yj in model
                        int row = trainData[idx].row;
                        int col = trainData[idx].col;
                        master_model.X.row(row) = Xi;
                        master_model.Y.col(col) = Yj;
                        cur_worker_size += 1;
                        //delay_counter[taskid - 1] = 1;
                        //cur_received_workers[taskid - 1] = 1;
                    
                        flag_receive = false;
                        //if (((cur_worker_size < FLAGS_group_size) || (max_element(delay_counter) > FLAGS_max_delay)) && (iter_counter < FLAGS_in_iters - 1)) 
                        //    flag_receive = true;
                        //if ((cur_worker_size < FLAGS_num_workers) && (iter_counter == FLAGS_in_iters - 1)) 
                        //    flag_receive = true;
                        //printf("master receive down\n");
                    //}
                }
            }
            // OutPut (epoch, acc, rmse)
            if (epoch % (FLAGS_n_epochs / numMaker) == 0){
                int trueNum = 0;
                long double error = 0;
                for (int i = 0; i < testExamples; i++){
                    mat xi = master_model.X.row(testData[i].row);
                    mat yj = master_model.Y.col(testData[i].col);
                    //xi.print("test xi: ");
                    mat predict = xi * yj;
                    if ( predict(0, 0) * testData[i].rating > 0 ) trueNum++;
                    error += pow(predict(0, 0) - testData[i].rating, 2);
                }
                double acc = double(trueNum) / testExamples;
                double rmse = sqrt(error) / testExamples;
                TrackOutput(epoch, acc, rmse, &stats);
                PrintOutput(epoch, acc, rmse);
            }    

        }
        return stats;
    }

};

#endif
