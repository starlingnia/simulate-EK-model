#include "setup.h"
#include <cmath>
#include <cstdlib>

using namespace std;

// ---------- Mat 基础 ----------
Mat::Mat() : N(0) {}
Mat::Mat(int n) : N(n), a(n*n, dcomplex(0,0)) {}
dcomplex& Mat::operator()(int i, int j){ return a[i*N + j]; }
const dcomplex& Mat::operator()(int i, int j) const { return a[i*N + j]; }

Mat Mat::identity(int n){
    Mat M(n);
    for(int i=0;i<n;i++) M(i,i) = dcomplex(1,0);
    return M;
}

// ---------- 运算 ----------
Mat mul(const Mat &A, const Mat &B){
    int N = A.N;
    Mat C(N);
    for(int i=0;i<N;i++){
        for(int k=0;k<N;k++){
            dcomplex aik = A(i,k);
            if(aik == dcomplex(0,0)) continue;
            for(int j=0;j<N;j++){
                C(i,j) += aik * B(k,j);
            }
        }
    }
    return C;
}

Mat dagger(const Mat &A){
    int N = A.N;
    Mat B(N);
    for(int i=0;i<N;i++) for(int j=0;j<N;j++) B(j,i) = conj(A(i,j));
    return B;
}

dcomplex trace(const Mat &A){
    dcomplex s(0,0);
    for(int i=0;i<A.N;i++) s += A(i,i);
    return s;
}

// ---------- EK模型 ----------
double full_action(const vector<Mat> &U, double beta){
    int D = (int)U.size();
    int N = U[0].N;
    double S = 0.0;
    for(int mu=0; mu<D; ++mu){
        for(int nu=mu+1; nu<D; ++nu){
            Mat tmp = mul(U[mu], U[nu]);
            Mat tmpd = dagger(tmp);
            dcomplex tr = trace(mul(tmp, tmpd));
            S += - beta * N * real(tr);
        }
    }
    return S;
}

double delta_action_for_update(const vector<Mat> &U, int mu, const Mat &U_mu_new, double beta){
    int D = (int)U.size();
    int N = U[0].N;
    double delta = 0.0;
    for(int nu=0; nu<D; ++nu){
        if(nu==mu) continue;
        Mat tmp_old = mul(U[mu], U[nu]);
        Mat tmp_new = mul(U_mu_new, U[nu]);
        Mat tmpd_old = dagger(tmp_old);
        Mat tmpd_new = dagger(tmp_new);
        double oldv = real(trace(mul(tmp_old, tmpd_old)));
        double newv = real(trace(mul(tmp_new, tmpd_new)));
        delta += - beta * N * (newv - oldv);
    }
    return delta;
}

// ---------- 工具 ----------
double rand_double(){ return (double)rand() / RAND_MAX; }
double gaussian_rand(double sigma=1.0){
    double u1 = rand_double();
    double u2 = rand_double();
    if(u1 < 1e-16) u1 = 1e-16;
    return sigma * sqrt(-2.0*log(u1)) * cos(2*M_PI*u2);
}

Mat random_hermitian(int N, double sigma){
    Mat H(N);
    for(int i=0;i<N;i++) H(i,i) = dcomplex(gaussian_rand(sigma),0.0);
    for(int i=0;i<N;i++){
        for(int j=i+1;j<N;j++){
            double re = gaussian_rand(sigma);
            double im = gaussian_rand(sigma);
            H(i,j) = dcomplex(re,im);
            H(j,i) = conj(H(i,j));
        }
    }
    return H;
}

Mat approx_exp_i_epsH(const Mat &H, double eps){
    int N = H.N;
    Mat I = Mat::identity(N);
    Mat H2 = mul(H,H);
    Mat R = I;
    for(int i=0;i<N*N;i++) R.a[i] += dcomplex(0, eps) * H.a[i];
    double coeff = -0.5*eps*eps;
    for(int i=0;i<N*N;i++) R.a[i] += coeff * H2.a[i];
    return R;
}

Mat left_mul_exp(const Mat &U, const Mat &H, double eps){
    Mat V = approx_exp_i_epsH(H, eps);
    return mul(V,U);
}

