#ifndef _TRAINER_
#define _TRAINER_

#include <iostream>
struct TrainStatistics{
    std::vector<int> epoch;
    std::vector<double> accuracy;
    std::vector<double> rmse;
};
typedef struct TrainStatistics TrainStatistics;

class Trainer {
protected:
    Model* model;
    Example* example;

    void TrackOutput(int cur_epoch, double cur_accuracy, double cur_rmse, TrainStatistics* stats){
        stats->epoch.push_back(cur_epoch);
        stats->accuracy.push_back(cur_accuracy);
        stats->rmse.push_back(cur_rmse);
    }
public:
    
    void PrintOutput(int cur_epoch, double cur_accuracy, double cur_rmse, double cur_time){
        //std::cout << "Epoch: " << cur_epoch << "    Accuracy: " << cur_accuracy << "    RMSE: " << cur_rmse << "    Spend Time: " << cur_time << " s" << std::endl;
        printf("Epoch: %d   Accuracy: %.4f  RMSE: %.4f  Spend Time: %.2f s \n", cur_epoch, cur_accuracy, cur_rmse, cur_time);
    }

    Trainer(Model* model, Example* example){
        this->model = model;
        this->example = example;
    }
    virtual ~Trainer(){}

    // Main training method
    virtual TrainStatistics Train(Model* model, Example* example, Updater* updater) = 0;
};


#endif
