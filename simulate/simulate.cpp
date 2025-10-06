#include "setup.h"
#include <iostream>
#include <ctime>

using namespace std;

int main(){
    srand(time(NULL));

    int N = 4, D = 4;
    double beta = 0.5, eps = 0.05;

    // 初始化 U_mu
    vector<Mat> U(D, Mat(N));
    for(int mu=0; mu<D; ++mu){
        U[mu] = Mat::identity(N);
    }

    // 简单做一个更新
    int mu = 0;
    Mat H = random_hermitian(N,1.0);
    Mat U_new = left_mul_exp(U[mu], H, eps);

    double dS = delta_action_for_update(U, mu, U_new, beta);
    cout << "ΔS = " << dS << endl;

    // 测试 full action
    double S = full_action(U, beta);
    cout << "S = " << S << endl;

    return 0;
}



