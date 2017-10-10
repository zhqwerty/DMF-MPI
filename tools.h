#include <chrono>
#include <iostream>

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


template <class T>
void printVec(const std::vector<T> nums){
    for (auto num : nums) std::cout << num << " ";
    std::cout << std::endl;
}

