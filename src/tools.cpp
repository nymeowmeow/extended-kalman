#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::cerr;
using std::endl;

const double EPS = 1e-4;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;
  //basic input checking
  if (estimations.empty())
  {
    cerr << "empty input" << endl;
    return rmse;
  }
  if (estimations.size() != ground_truth.size())
  {
    cerr << "Input size mismatch between estimation and ground truth" << endl;
    return rmse;
  }
  for (unsigned int i = 0; i < estimations.size(); ++i)
  {
      VectorXd residual = estimations[i] - ground_truth[i];
      residual = residual.array()*residual.array();
      rmse += residual;
  }
  //calculate root mean square of the residual
  rmse = rmse/estimations.size();
  rmse = rmse.array().sqrt();
  return rmse; 
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  double px = x_state(0);
  double py = x_state(1);
  double vx = x_state(2);
  double vy = x_state(3);

  MatrixXd Hj(3,4);
  double c1 = px*px + py*py;
  //c1 = std::max(c1, EPS); //make sure we don't divide by 0
  if (c1 < EPS)
  {
      //return Hj.setZero();
      c1 = EPS;
  } 
  double c2 = sqrt(c1);
  double c3 = c1*c2;
  Hj << (px/c2), (py/c2), 0, 0,
       -(py/c1), (px/c1), 0, 0,
        py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;
  return Hj;
}

double Tools::adjustRange(double theta) {
  if (theta < -M_PI)
  {
    while (theta < -M_PI)
    { 
      theta += 2*M_PI;
    }
  } else if (theta >= M_PI) {
    while (theta > M_PI)
    {
      theta -= 2*M_PI;
    }
  }
  return theta;
}

