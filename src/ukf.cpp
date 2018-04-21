#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);
  x_.setZero();

  // initial covariance matrix
  P_ = MatrixXd(5, 5);
  P_.setIdentity();

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 2.0;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI/4;
  
  //DO NOT MODIFY measurement noise values below these are provided by the sensor manufacturer.
  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  //DO NOT MODIFY measurement noise values above these are provided by the sensor manufacturer.
  
  // initialization flag
  is_initialized_ = false;
  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3-n_aug_;
  time_us_ = 0;
  Xsig_pred_.setZero(n_x_, 2*n_aug_+1);
  weights_.setZero(n_x_);
  weights_aug_.setZero(2*n_aug_+1);

  //measurement covariance matrix - laser
  R_laser_ = MatrixXd(2, 2);
  R_laser_ << std_laspx_*std_laspx_, 0,
              0, std_laspy_*std_laspy_;

  //measurement covariance matrix - radar
  R_radar_ = MatrixXd(3, 3);
  R_radar_ << std_radr_*std_radr_, 0, 0,
              0, std_radphi_*std_radphi_, 0,
              0, 0, std_radrd_*std_radrd_;

  ///* Count of NIS params 
  // NIS_lidar <5%, %5-95%, >%95 
  // NIS_radar <5%, %5-95%, >%95 
  NIS_hist.setZero(2,3);
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  // check if initialized
  if (!is_initialized_) {
    // initialize x_ and P_
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    /**
    Convert radar from polar to cartesian coordinates and initialize state.
    */
    cout << "first data is radar" << endl;
    double rho, theta;
    rho = meas_package.raw_measurements_[0];
    theta = meas_package.raw_measurements_[1];
    x_(0) = rho*cos(theta);
    x_(1) = rho*sin(theta);
    cout << "radar initialized" << endl;
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    /**
    Initialize state.
    */
    cout << "first data is laser" << endl;
    x_(0) = meas_package.raw_measurements_[0];
    x_(1) = meas_package.raw_measurements_[1];
    cout << "laser initialized" << endl;
    }

    // done initializing, no need to predict or update
    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    // print the output
    cout << "x_ = " << x_ << endl;
    cout << "P_ = " << P_ << endl;
    return;
  }
  
  // skip not used sensor
  if ( (use_laser_ == false && meas_package.sensor_type_ == MeasurementPackage::LASER) || (use_radar_ == false && meas_package.sensor_type_ == MeasurementPackage::RADAR))
    return;

  // already initialized
  //prediction using CVTR model
  double dt = (meas_package.timestamp_ - time_us_)/1000000.0;
  time_us_ = meas_package.timestamp_;

  // ekf prediction, radar and lidar are the same
  Prediction(dt);

  // update prediction for radar and lidar
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    UpdateRadar(meas_package);
  } else {
    // Laser updates
    UpdateLidar(meas_package);
  }

  // print the output
  cout << "x_ = " << x_.transpose() << endl;
  //cout << "P_ = " << P_ << endl;

  // print NIS percentale
  double total = NIS_hist(0,0) + NIS_hist(0,1) + NIS_hist(0,2);
  cout << "NIS_lidar percentage (nominal 5%,90%,5%): " << NIS_hist(0,0)/total<<", " << NIS_hist(0,1)/total<<", "<< NIS_hist(0,2)/total<<endl;
  total = NIS_hist(1,0) + NIS_hist(1,1) + NIS_hist(1,2);
  cout << "NIS_radar percentage (nominal 5%,90%,5%): " << NIS_hist(1,0)/total<<", " << NIS_hist(1,1)/total<<", "<< NIS_hist(1,2)/total<<endl;
}


/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  //cout<< "Step 1: Prediction step started"<<endl;
  //create augmented mean vector
  VectorXd x_aug = VectorXd(7);
  x_aug.setZero();

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);
  P_aug.setZero();

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  //create augmented mean state
  x_aug.head(5) = x_;

  //create augmented covariance matrix
  MatrixXd Q(2,2);
  Q << std_a_*std_a_, 0,
       0, std_yawdd_*std_yawdd_;
  P_aug.topLeftCorner(5,5) = P_;
  P_aug.bottomRightCorner(2,2) = Q;
  //create square root matrix
  MatrixXd A = P_aug.llt().matrixL();
  //create augmented sigma points
  MatrixXd xM(7,7);
  VectorXd onev(7);
  onev.setOnes();
  xM = x_aug*onev.transpose();
  Xsig_aug.col(0) = x_aug;
  Xsig_aug.block<7,7>(0,1) = xM + sqrt(lambda_ + n_aug_)*A;
  Xsig_aug.block<7,7>(0,8) = xM - sqrt(lambda_ + n_aug_)*A;

  //write predicted sigma points into right column
  for (int i =0; i<Xsig_aug.cols(); i++){
      VectorXd xk(n_aug_);
      xk = Xsig_aug.col(i);
      VectorXd xnew(5);
      xnew = xk.head(5);
      if (xk(4)>1e-6){
          xnew(0) += xk(2)/xk(4)*(sin(xk(3)+xk(4)*delta_t)-sin(xk(3))) + 0.5*delta_t*delta_t*cos(xk(3))*xk(5);
          xnew(1) += xk(2)/xk(4)*(-cos(xk(3)+xk(4)*delta_t)+cos(xk(3))) + 0.5*delta_t*delta_t*sin(xk(3))*xk(5);
          xnew(2) += 0 + delta_t*xk(5);
          xnew(3) += xk(4)*delta_t + 0.5*delta_t*delta_t*xk(6);
          xnew(4) += 0 + delta_t*xk(6);
      }
      else{
          xnew(0) += xk(2)*cos(xk(3))*delta_t + 0.5*delta_t*delta_t*cos(xk(3))*xk(5);
          xnew(1) += xk(2)*sin(xk(3))*delta_t + 0.5*delta_t*delta_t*sin(xk(3))*xk(5);
          xnew(2) += 0 + delta_t*xk(5);
          xnew(3) += xk(4)*delta_t + 0.5*delta_t*delta_t*xk(6);
          xnew(4) += 0 + delta_t*xk(6);
      }
      Xsig_pred_.col(i) = xnew;
  }

  // get mean and variance of prediction from sigma points
  //set weights
  weights_aug_(0) = lambda_/(lambda_+n_aug_);
  weights_aug_.tail(2*n_aug_).setConstant(0.5/(lambda_+n_aug_));
  //predict state mean
  x_ = Xsig_pred_*weights_aug_;
  //angle normalization
  while (x_(3)> M_PI) x_(3)-=2.*M_PI;
  while (x_(3)<-M_PI) x_(3)+=2.*M_PI;
  P_.setZero();
  //predict state covariance matrix
  for (int i=0; i<2*n_aug_+1; i++){
      VectorXd xdiff;
      xdiff = Xsig_pred_.col(i)-x_;
      //angle normalization
      while (xdiff(3)> M_PI) xdiff(3)-=2.*M_PI;
      while (xdiff(3)<-M_PI) xdiff(3)+=2.*M_PI;
      P_ += weights_aug_(i)*xdiff*xdiff.transpose();
  }
  //cout<< "Step 1: Prediction step finished"<<endl;
  cout << "x_ = " << x_.transpose() << endl;
  //cout << "P_ = " << P_ << endl;
}




/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  //cout<< "Step 2: Update laser measurement started"<<endl;
  //create matrix for sigma points in measurement space
  int n_z = 2;
  MatrixXd Zsig(n_z, 2 * n_aug_ + 1);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.setZero();

  //transform sigma points into measurement space
  //calculate mean predicted measurement
  //calculate innovation covariance matrix S
  for (int i=0; i<2 * n_aug_ + 1; i++){
      Zsig(0,i) = Xsig_pred_(0,i);
      Zsig(1,i) = Xsig_pred_(1,i);
  }

  // predict measurement mean
  z_pred = Zsig*weights_aug_;
  
  // predict measurement variance
  for (int i=0; i<2 * n_aug_ + 1; i++){
      VectorXd diff = Zsig.col(i) - z_pred;
      S += weights_aug_(i)*diff*diff.transpose();
  }
  S += R_laser_;

  //create example vector for incoming radar measurement
  VectorXd z = VectorXd(n_z);
  z << meas_package.raw_measurements_;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.setZero();

  //calculate cross correlation matrix
  //calculate Kalman gain K;
  //update state mean and covariance matrix
  for (int i=0; i<2 * n_aug_ + 1; i++){
      VectorXd xi, zi;
      xi = Xsig_pred_.col(i) - x_;
      zi = Zsig.col(i) - z_pred;
      while (xi(3)>M_PI) xi(3)-=2*M_PI;
      while (xi(3)<-M_PI) xi(3)+=2*M_PI;
      Tc += weights_aug_(i)*xi*zi.transpose();
  }

  MatrixXd K;
  K = Tc*S.inverse();
  
  x_ = x_ + K*(z - z_pred);
  P_ = P_ - K*S*K.transpose();

  //cout<< "Step 2: Update lidar measurement finished"<<endl;

  // calculate the lidar NIS
  // 2df, %95 above 0.103, 95% below 5.991; 
  double NIS;
  NIS = (z - z_pred).transpose()*S.inverse()*(z - z_pred);
  if (NIS<0.103)
    NIS_hist(0, 0)+=1;
  else if (NIS>5.991)
    NIS_hist(0, 2)+=1;
  else
    NIS_hist(0, 1)+=1;
  //cout<<"NIS_lidar = "<<NIS<<endl;
}




/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  //cout<< "Step 2: Update radar measurement started"<<endl;
  //create matrix for sigma points in measurement space
  int n_z = 3;
  MatrixXd Zsig(n_z, 2 * n_aug_ + 1);

  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z,n_z);
  S.setZero();

  //transform sigma points into measurement space
  //calculate mean predicted measurement
  //calculate innovation covariance matrix S
  for (int i=0; i<2 * n_aug_ + 1; i++){
      Zsig(0,i) = sqrt(Xsig_pred_(0,i)*Xsig_pred_(0,i) + Xsig_pred_(1,i)*Xsig_pred_(1,i) );
      Zsig(1,i) = atan2(Xsig_pred_(1,i), Xsig_pred_(0,i));
      if (Zsig(0,i)<1e-6)
        Zsig(2,i) = 0;
      else
        Zsig(2,i) = (Xsig_pred_(0,i)*cos(Xsig_pred_(3,i))*Xsig_pred_(2,i)+Xsig_pred_(1,i)*sin(Xsig_pred_(3,i))*Xsig_pred_(2,i))/Zsig(0,i);
  }

  // predict measurement mean
  z_pred = Zsig*weights_aug_;
  
  // predict measurement variance
  for (int i=0; i<2 * n_aug_ + 1; i++){
      VectorXd diff = Zsig.col(i) - z_pred;
      while(diff(1)>M_PI) diff(1) -= 2*M_PI;
      while(diff(1)<-M_PI) diff(1) += 2*M_PI;
      S += weights_aug_(i)*diff*diff.transpose();
  }
  S += R_radar_;

  //create example vector for incoming radar measurement
  VectorXd z = VectorXd(n_z);
  z << meas_package.raw_measurements_;

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.setZero();

  //calculate cross correlation matrix
  //calculate Kalman gain K;
  //update state mean and covariance matrix
  for (int i=0; i<2 * n_aug_ + 1; i++){
      VectorXd xi, zi;
      xi = Xsig_pred_.col(i) - x_;
      zi = Zsig.col(i) - z_pred;
      while (xi(3)>M_PI) xi(3)-=2*M_PI;
      while (xi(3)<-M_PI) xi(3)+=2*M_PI;
      while (zi(1)>M_PI) zi(1)-=2*M_PI;
      while (zi(1)<-M_PI) zi(1)+=2*M_PI;
      Tc += weights_aug_(i)*xi*zi.transpose();
  }

  MatrixXd K;
  K = Tc*S.inverse();
  
  x_ = x_ + K*(z - z_pred);
  P_ = P_ - K*S*K.transpose();

  //cout<< "Step 2: Update radar measurement finished"<<endl;

  // calculate the radar NIS
  // 3df, 95% above 0.352, 95% below 7.815;
  double NIS;
  NIS = (z - z_pred).transpose()*S.inverse()*(z - z_pred);
  if (NIS<0.352)
    NIS_hist(1, 0)+=1;
  else if (NIS>7.815)
    NIS_hist(1, 2)+=1;
  else
    NIS_hist(1, 1)+=1;
  //cout<<"NIS_radar = "<<NIS<<endl;
}
