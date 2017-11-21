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
    int FLAGS_in_iters = 1e5;
    int FLAGS_num_workers = 2;
    bool FLAGS_Asy = false;
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
        Timer timer;
        timer.Tick();
        for (int epoch = 0; epoch < FLAGS_n_epochs; epoch++){
            double step = FLAGS_Asy ? 0.1 : 1;
            double learning_rate = step / std::pow(1+epoch, 0.1);
            //std::cout << "Epoch: " << epoch << std::endl;

            for(int iter_counter = 0; iter_counter < FLAGS_in_iters; iter_counter++){
                // build message : Xi, Yj, idx, flag_break
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
                    if (iter_counter < FLAGS_in_iters - 1 || epoch < FLAGS_n_epochs - 1) message.push_back(0);
                    else message.push_back(1);
                    
                    // send messages to workers
                    MPI_Send(&message[0], model->rank * 2 + 2, MPI_DOUBLE, i, 102, MPI_COMM_WORLD);
                }
                
                // Receive info(gradXi, gradYj, idx) and update(ApplyGradient)
                std::vector<double> info(model->rank * 2 + 1, 0);
                if (FLAGS_Asy){
                    MPI_Probe(MPI_ANY_SOURCE, 101, MPI_COMM_WORLD, &status);    
                    int taskid = status.MPI_SOURCE;
                    MPI_Recv(&info[0], model->rank * 2 + 1, MPI_DOUBLE, taskid, 101, MPI_COMM_WORLD, &status);
                    
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
                else{
                    for (int iter = 0; iter < FLAGS_num_workers; iter++){
                        MPI_Probe(MPI_ANY_SOURCE, 101, MPI_COMM_WORLD, &status);    
                        int taskid = status.MPI_SOURCE;
                        MPI_Recv(&info[0], model->rank * 2 + 1, MPI_DOUBLE, taskid, 101, MPI_COMM_WORLD, &status);

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
