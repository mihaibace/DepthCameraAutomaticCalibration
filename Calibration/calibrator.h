#ifndef _CALIBRATOR_H
#define _CALIBRATOR_H

#include <opencv2/core/core.hpp>

#include <vector>
#include <map>

namespace aptarism {
namespace vision {

class Calibrator {
public:
  
  void addProjCam(float x, float y) {
      projections_.push_back(cv::Point2f(x, y));
  }
  void add3DPoint(float x, float y, float depth) {
      points3D_.push_back(cv::Point3f(x, y, depth));
  }

  typedef std::vector<cv::Point2f> Point2DVector;
  typedef std::vector<cv::Point3f> Point3DVector;

  const Point2DVector& projections() { return projections_; }
  const Point3DVector& points3D() const { return points3D_; }

  cv::Mat calibrate() const;
  void save() const;
  cv::Mat load();
  unsigned numEntries() const;

private:
  cv::Mat reprojDifference(const cv::Mat& current) const;
  cv::Mat jacobianFunc(const double a[12]) const;
  cv::Mat evalFunc(const double a[12]) const;

  Point2DVector projections_;
  Point3DVector points3D_;
};

}  // namespace vision
}  // namespace aptarism

#endif  // _CALIBRATOR_H
