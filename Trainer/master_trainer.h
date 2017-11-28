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
    int FLAGS_n_epochs = 30;
    int FLAGS_in_iters = 1e4;
    int FLAGS_num_workers = 8;
    int FLAGS_max_delay = 4;
    bool FLAGS_Asy = true;
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
        int flag_asy = FLAGS_Asy ? 1 : 0;
        printf("Number of Workers: %d,   FLAGS_Asy: %d\n", FLAGS_num_workers, flag_asy);
        std::cout << "Start Training" << std::endl;
        
        std::vector<int> cur_received_workers(FLAGS_num_workers, 1);

        Timer timer;
        timer.Tick();
        for (int epoch = 0; epoch < FLAGS_n_epochs; epoch++){
            double step = FLAGS_Asy ? 1 : 10;
            std::vector<int> delay_counter(FLAGS_num_workers, 1);
           
            //double learning_rate = step / std::pow(1+epoch, 0.1);
            //std::cout << "Epoch: " << epoch << std::endl;
            for(int iter_counter = 0; iter_counter < FLAGS_in_iters; iter_counter++){
                double learning_rate = step / std::pow(1 + epoch * FLAGS_in_iters + iter_counter, 0.1);
                // build message : Xi, Yj, idx, flag_break
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
                    if (iter_counter < FLAGS_in_iters - 1 || epoch < FLAGS_n_epochs - 1) message.push_back(0);
                    else if (iter_counter == FLAGS_in_iters - 1 && epoch == FLAGS_n_epochs - 1) message.push_back(1);
                    else message.push_back(1);
                    
                    // send messages to workers
                    if (cur_received_workers[i] == 0){
                        delay_counter[i] += 1;
                    }
                    else{ 
                        MPI_Send(&message[0], model->rank * 2 + 2, MPI_DOUBLE, i + 1, 102, MPI_COMM_WORLD);
                        //std::cout << "Sent to worker: " << i + 1 << std::endl;
                    }
                }
                
                cur_received_workers.assign(FLAGS_num_workers, 0);   // clear receive list
                // Receive info(gradXi, gradYj, idx) and update(ApplyGradient)
                std::vector<double> info(model->rank * 2 + 1, 0);
                int cur_worker_size = 0;
                if (FLAGS_Asy){
                    bool flag_receive = true;
                    while (flag_receive){

                        MPI_Probe(MPI_ANY_SOURCE, 101, MPI_COMM_WORLD, &status);    
                        int taskid = status.MPI_SOURCE;
                        //std::cout << "Receive taskid: " << taskid << std::endl;
                        MPI_Recv(&info[0], model->rank * 2 + 1, MPI_DOUBLE, taskid, 101, MPI_COMM_WORLD, &status);

                        cur_received_workers[taskid - 1] = 1; // set reveiced worker to be 1;
                        delay_counter[taskid - 1] = 1;
                        cur_worker_size += 1;

                        // Prase updaterd_Xi_Yj(Xi, Yj, idx);
                        // convert vector to mat
                        std::vector<double> gradXi_tmp(info.begin(), info.begin() + model->rank);
                        std::vector<double> gradYj_tmp(info.begin() + model->rank, info.begin() + model->rank * 2);
                        mat gradXi = vec_2_mat(gradXi_tmp, 0, 1, model->rank);
                        mat gradYj = vec_2_mat(gradYj_tmp, 0, model->rank, 1);
                        int idx = *(info.end() - 1);
                    
                        // update Xi, Yj in model
                        int row = trainData[idx].row;
                        int col = trainData[idx].col;

                        updater->ApplyGradient(master_model, row, col, gradXi, gradYj, learning_rate);
                        
                        flag_receive = false;
                        //std::cout << "cur_worker_size: " << cur_worker_size << std::endl;
                        
                        // Bounded Delay   max_delay = 1  => SYN; max_delay = inf => ASY.
                        if (max_element(delay_counter) > FLAGS_max_delay && iter_counter < FLAGS_in_iters - 1 && epoch < FLAGS_n_epochs - 1){
                            //std::cout << " delay " << std::endl;
                            flag_receive = true;
                        }
                        // Delay for last iteration
                        if (cur_worker_size < FLAGS_num_workers && epoch == FLAGS_n_epochs - 1 && iter_counter == FLAGS_in_iters - 1){ 
                            //std::cout << "2222" << std::endl;
                            flag_receive = true;
                        }
                        //std::cout  << "max_element: " << max_element(delay_counter) << "    delay_counter : " << std::endl;
                        //printVec(delay_counter);
                    }
                }
                else{
                    for (int iter = 0; iter < FLAGS_num_workers; iter++){
                        MPI_Probe(MPI_ANY_SOURCE, 101, MPI_COMM_WORLD, &status);    
                        int taskid = status.MPI_SOURCE;
                       // std::cout << "Receive taskid: " << taskid << std::endl;
                        MPI_Recv(&info[0], model->rank * 2 + 1, MPI_DOUBLE, taskid, 101, MPI_COMM_WORLD, &status);

                        cur_received_workers[taskid - 1] = 1; // set reveiced worker to be 1;
                        // Prase updaterd_Xi_Yj(Xi, Yj, idx);
                        // convert vector to mat
                        std::vector<double> gradXi_tmp(info.begin(), info.begin() + model->rank);
                        std::vector<double> gradYj_tmp(info.begin() + model->rank, info.begin() + model->rank * 2);
                        mat gradXi = vec_2_mat(gradXi_tmp, 0, 1, model->rank);
                        mat gradYj = vec_2_mat(gradYj_tmp, 0, model->rank, 1);
                        int idx = *(info.end() - 1);
                    
                        // update Xi, Yj in model
                        int row = trainData[idx].row;
                        int col = trainData[idx].col;

                        updater->ApplyGradient(master_model, row, col, gradXi, gradYj, learning_rate);
                    }
                }
            }
            // OutPut (epoch, acc, rmse)
            int trueNum = 0;
            long double error = 0;
            for (int i = 0; i < testExamples; i++){
                mat xi = master_model.X.row(testData[i].row);
                mat yj = master_model.Y.col(testData[i].col);
                //xi.print("test xi: ");
                mat predict = xi * yj;
                if ( sign(predict(0, 0)) == sign(testData[i].rating)) trueNum++;
                error += pow(predict(0, 0) - testData[i].rating, 2);
            }
            double acc = double(trueNum) / testExamples;
            double rmse = sqrt(error / testExamples);
            timer.Tock();
            TrackOutput(epoch + 1, acc, rmse, timer.duration, &stats);
            PrintOutput(epoch + 1, acc, rmse, timer.duration);
        }
        return stats;
    }
};

#endif
