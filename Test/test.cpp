#include <iostream>
#include <armadillo>
using namespace arma;

int main(){
    mat X = randn<mat>(10, 10);
    X.print();
    return 0;
}