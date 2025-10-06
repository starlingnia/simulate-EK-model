#pragma once
#include <complex>
#include <vector>

// 简化类型
using dcomplex = std::complex<double>;

struct Mat {
    int N;
    std::vector<dcomplex> a;
    Mat();
    Mat(int n);

    dcomplex& operator()(int i, int j);
    const dcomplex& operator()(int i, int j) const;

    static Mat identity(int n);
};

// 线性代数运算
Mat mul(const Mat &A, const Mat &B);
Mat dagger(const Mat &A);
dcomplex trace(const Mat &A);

// EK模型相关
double full_action(const std::vector<Mat> &U, double beta);
double delta_action_for_update(const std::vector<Mat> &U, int mu, const Mat &U_mu_new, double beta);

// 工具
Mat random_hermitian(int N, double sigma);
Mat approx_exp_i_epsH(const Mat &H, double eps);
Mat left_mul_exp(const Mat &U, const Mat &H, double eps);
