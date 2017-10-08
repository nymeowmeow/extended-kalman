#include "kalman_filter.h"
#include <cmath>
#include <iostream>
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}
const double EPS = 1e-4;

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  P_ = F_*P_*F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd y = z - H_*x_;
  updateState(y);
}

void KalmanFilter::updateState(const VectorXd& y) {
  MatrixXd Ht = H_.transpose();
  MatrixXd S  = H_ * P_ * Ht + R_;
  MatrixXd K = P_ * Ht * S.inverse();
  //new estimate
  x_ = x_ + K*y;
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K*H_)*P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  double px = x_(0);
  double py = x_(1);
  double vx = x_(2);
  double vy = x_(3);

  if (px > -EPS && px < EPS)
    px = (px > 0)?EPS:-EPS;

  double rho = sqrt(px*px + py*py);
  rho = std::max(rho, EPS);
  double theta = atan2(py, px);
  //std::cout << "pi is: " << M_PI << std::endl;
  //theta = adjustRange(theta);
  //make sure it is in the range -pi to pi
  double rho_dot = (px*vx + py*vy)/rho;

  VectorXd h = VectorXd(3);
  h << rho, theta, rho_dot;
  VectorXd y = z - h;
  theta = Tools::adjustRange(y(1));
  y(1) = theta;
  updateState(y);
}
