#fndef _TRAINER_
#define _TRAINER_

#include <iostream>
struct TrainsStatistics{
    std::vector<int> epoch;
    std::vector<double> accuracy;
    std::vector<double> rmse;
}
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
    
    void PrintOutput(int cur_epoch, double cur_accuracy, double cur_rmse){
        std::cout << "Epoch: " << cur_epoch << " Accuracy: " << cur_accuracy << " RMSE: " << cur_rmse << std::endl;
    }

    Trainer(Model* model, Example* example){
        this->model = model;
        this->example = example;
    }
    virtual ~Trainer(){}

    // Main training method
    virtual TrainStatistics Train(Model* model, Example* example, Upddater* updater) = 0;
};


#endif
