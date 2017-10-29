#ifndef _TOOLS_
#define _TOOLS_

#include <chrono>
#include <iostream>
#include <armadillo>
using namespace arma;

class Timer
{
    typedef std::chrono::steady_clock::time_point   tp;
    typedef std::chrono::duration<double>           dd;
    typedef std::chrono::steady_clock               sc;
public:
    tp start;
    tp end;
    dd span;
    double duration;
    Timer(): start(sc::now()), span(dd(0)){}
    void Tick(){
        start = sc::now();
    }
    void Tock(){
        end = sc::now();
        span += std::chrono::duration_cast<dd>(end - start);
        duration = span.count();
        span = dd(0);
    }
};

template<typename T>
std::vector<double> mat_2_vec(const T& Matrix){
  int row = Matrix.n_rows;
  int col = Matrix.n_cols;
  std::vector<double> out(row*col, 0);
  for(int i=0; i<row; i++)
    for(int j=0; j<col; j++)
      out[i*col+j] = Matrix(i,j);

  return out;
}


mat vec_2_mat(const std::vector<double>& w, int begin, int row, int col){
  mat out(row,col); int index = 0;
    for(int i=0; i<row; i++)
      for(int j=0; j<col; j++){
        index = i*col + j;
        out(i,j) = w[begin+index];
      }
  return out;
}

template <class T>
void printVec(const std::vector<T> nums){
    for (auto num : nums) std::cout << num << " ";
    std::cout << std::endl;
}

#endif
