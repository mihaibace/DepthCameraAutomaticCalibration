#include "calibrator.h"

#include <assert.h>
#include <algorithm>
#include <stdio.h>

using std::vector;

namespace aptarism {
namespace vision {

// This is the inner function to minimize, for each clicked point.
void reprojFunc(const double a[12], double u, double v, double z, double r[2]) {
	const double a11 = a[0];
	const double a12 = a[1];
	const double a13 = a[2];
	const double a14 = a[3];
	const double a21 = a[4];
	const double a22 = a[5];
	const double a23 = a[6];
	const double a24 = a[7];
	const double a31 = a[8];
	const double a32 = a[9];
	const double a33 = a[10];
	const double a34 = a[11];

    const double t1 = a11 * u * z  + a12 * v * z + a13 * z + a14;
    const double t2 = a21 * u * z  + a22 * v * z + a23 * z + a24;
    const double t3 = a31 * u * z  + a32 * v * z + a33 * z + a34;
    r[0] = t1 / t3;
    r[1] = t2 / t3;
}

// The jacobian of the above function.
void reprojJacobian(const double a[12], double u, double v, double z, double J0[12], double J1[12]) {
	const double a11 = a[0];
	const double a12 = a[1];
	const double a13 = a[2];
	const double a14 = a[3];
	const double a21 = a[4];
	const double a22 = a[5];
	const double a23 = a[6];
	const double a24 = a[7];
	const double a31 = a[8];
	const double a32 = a[9];
	const double a33 = a[10];
	const double a34 = a[11];

    const double t1 = (a31*u*z + a32*v*z + a33*z + a34);
    J0[0] = u*z/t1;
    J0[1] = v*z/t1;
    J0[2] = z/t1;
    J0[3] = 1.0/t1;
    J0[4] = 0;
    J0[5] = 0;
    J0[6] = 0;
    J0[7] = 0;
    J0[8] = -(a11*u*z + a12*v*z + a13*z + a14)*u*z/t1/t1;
    J0[9] = -(a11*u*z + a12*v*z + a13*z + a14)*v*z/t1/t1;
    J0[10] = -(a11*u*z + a12*v*z + a13*z + a14)*z/t1/t1;
    J0[11] = -(a11*u*z + a12*v*z + a13*z + a14)/t1/t1;
    J1[0] = 0;
    J1[1] = 0;
    J1[2] = 0;
    J1[3] = 0;
    J1[4] = u*z/t1;
    J1[5] = v*z/t1;
    J1[6] = z/t1;
    J1[7] = 1/t1;
    J1[8] = -(a21*u*z + a22*v*z + a23*z + a24)*u*z/t1/t1;
    J1[9] = -(a21*u*z + a22*v*z + a23*z + a24)*v*z/t1/t1;
    J1[10] = -(a21*u*z + a22*v*z + a23*z + a24)*z/t1/t1;
    J1[11] = -(a21*u*z + a22*v*z + a23*z + a24)/t1/t1;
}

// The function to optimize. Calls reprojFunc for each point.
cv::Mat Calibrator::evalFunc(const double a[12]) const {
	assert(numEntries() > 6);
	cv::Mat reproj(numEntries(), 2, CV_64FC1);

	for (unsigned i = 0; i < numEntries(); ++i) {
		reprojFunc(a, points3D_[i].x, points3D_[i].y, points3D_[i].z,
			reproj.ptr<double>(i));
	}
    return reproj.reshape(1, 2 * numEntries());
}

// The global jacobian function.
cv::Mat Calibrator::jacobianFunc(const double a[12]) const {
	assert(numEntries() > 6);
	cv::Mat jacobian(numEntries() * 2, 12, CV_64FC1);

	for (unsigned i = 0; i < numEntries(); ++i) {
		reprojJacobian(a, points3D_[i].x, points3D_[i].y, points3D_[i].z,
			jacobian.ptr<double>(i * 2),
			jacobian.ptr<double>(i * 2 + 1));
	}
    return jacobian;
}

cv::Mat Calibrator::reprojDifference(const cv::Mat& current) const {
    cv::Mat target;
    cv::Mat(projections_).convertTo(target, CV_64FC1);

	cv::Mat delta = target.reshape(1, numEntries() * 2) - current;
    return delta;
}

// Levenberg-Marquardt to optimize the objective function.
cv::Mat Calibrator::calibrate() const {
    double a[12] = {
        3, 0, 100, 1,
        0, 100, 100, 1,
		0, 0, 1, 1};


    cv::Mat A(12, 1, CV_64FC1, a);

    if (numEntries() <= 6)
        return A;

	cv::Mat delta(12, 1, CV_64FC1);
	delta = cv::Scalar::all(0);
	double lambda = 1;
	double prevErr = 1e50;
    for (int i = 0; i < 200; ++i) {
		cv::Mat ATest = A + lambda*delta;
		//ATest *= 100.0 / cv::norm(ATest.reshape(1, 3).col(3));

        cv::Mat current = evalFunc(ATest.ptr<double>(0));
        cv::Mat err = reprojDifference(current);
		double reprojError = cv::norm(err);
		if (1) {
			printf("Reproj error: %f lambda: %f\n", reprojError, lambda);
		}
        cv::Mat J = jacobianFunc(ATest.ptr<double>(0));

		if (reprojError > prevErr || !cv::solve(J, err, delta, cv::DECOMP_NORMAL | cv::DECOMP_LU )) {
			if (0) {
				printf("backtracking prev err: %f, new: %f, lambda = %f\n", prevErr, reprojError, lambda);
			}
            lambda /= 10.0;
			if (lambda < 1e-10) {
				break;
			}
		} else {
			A = ATest;
			prevErr = reprojError;
			lambda *= 2;
			if (0)
			printf("Delta: %f %f %f %f %f %f %f %f %f %f %f %f\n",
					delta.at<double>(0, 0),
					delta.at<double>(1, 0),
					delta.at<double>(2, 0),
					delta.at<double>(3, 0),
					delta.at<double>(4, 0),
					delta.at<double>(5, 0),
					delta.at<double>(6, 0),
					delta.at<double>(7, 0),
					delta.at<double>(8, 0),
					delta.at<double>(9, 0),
					delta.at<double>(10, 0),
					delta.at<double>(11, 0));
		}
    }
	printf("A: %f %f %f %f %f %f %f %f %f %f %f %f\nErr: %f",
                   A.at<double>(0, 0),
                   A.at<double>(1, 0),
                   A.at<double>(2, 0),
                   A.at<double>(3, 0),
                   A.at<double>(4, 0),
                   A.at<double>(5, 0),
                   A.at<double>(6, 0),
                   A.at<double>(7, 0),
                   A.at<double>(8, 0),
                   A.at<double>(9, 0),
                   A.at<double>(10, 0),
                   A.at<double>(11, 0),
				   sqrt(prevErr));

    return A.clone();
}

unsigned Calibrator::numEntries() const {
	return std::min(projections_.size(), points3D_.size());
}

void Calibrator::save() const {
	if (numEntries() == 0) {
		return;
	}

	cv::Mat A = calibrate();
    cv::FileStorage fs("calib_clicks.yml", cv::FileStorage::WRITE);
    fs << "features" << "[";
    for (unsigned int i = 0; i < numEntries(); ++i) {
        fs << "{:" << "px" << projections_[i].x << "py" << projections_[i].y
            << "u" << points3D_[i].x << "v" << points3D_[i].y << "z" << points3D_[i].z << "}";
    }
    fs << "]";

    fs << "A" << A;
    fs.release();
}

cv::Mat Calibrator::load() {
    cv::FileStorage fs("calib_clicks.yml", cv::FileStorage::READ);
	if (!fs.isOpened()) {
		return cv::Mat();
	}
    cv::FileNode features = fs["features"];
	for (cv::FileNodeIterator it = features.begin(); it != features.end(); ++it) {
        addProjCam((*it)["px"], (*it)["py"]);
        add3DPoint((*it)["u"], (*it)["v"], (*it)["z"]);
    }
	cv::Mat A;
	fs["A"] >> A;
    fs.release();
	//calibrate();
	return A;
}

}  // namespace vision
}  // namespace aptarism
