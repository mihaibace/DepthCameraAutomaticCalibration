#include <Windows.h>
#include <Ole2.h>

#include <iostream>
#include <fstream>

#include <gl/GL.h>
#include <gl/GLU.h>
#include <gl/glut.h>

#include <NuiApi.h>
#include <NuiImageCamera.h>
#include <NuiSensor.h>

#include <assert.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

#include <opencv2/imgproc/imgproc.hpp>  // Gaussian Blur
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat, Scalar)
#include <opencv2/highgui/highgui.hpp>  // OpenCV window I/O

#include "calibrator.h"

using namespace cv;
using namespace std;
using namespace aptarism::vision;

namespace
{
	const int width = 640;
	const int height = 480;

	// Kinect variables
	HANDLE depthStream;				// The indetifier of the Kinect's Depth Camera
	HANDLE rgbStream;				// The identifier of the Kinect's RGB Camera
	INuiSensor * sensor;            // The kinect sensor
	NUI_IMAGE_FRAME imageRGBFrame;
	BYTE rgbData[width*height*4];

	// Webcam variables
	CvCapture* capture;

	// Calibrator 
	Calibrator webcamCalibrator;
	Calibrator kinectCalibrator;
	
	cv::Mat webcam_calib_result;
	cv::Mat kinect_calib_result;

	double webcam_a_calib[12];
	double kinect_a_calib[12];

	// Define minimum number of calibration points
	unsigned webcamMinCalibrationPoints = 30;
	unsigned kinectMinCalibrationPoints = 30;

	// Template matching
	cv::Mat rgb_template_1;
	cv::Mat kinect_rgb_template_1;
	cv::Mat kinect_depth_template_1;
	cv::Mat rgb_template_2;
	cv::Mat kinect_rgb_template_2;
	cv::Mat kinect_depth_template_2;

	// Frame skipping variable
	int frameCount = 0;

	bool webcamCalibrated = false;
	bool kinectCalibrated = false;

	typedef std::vector<cv::Point2f> Point2DVector;
	typedef std::vector<cv::Point3f> Point3DVector; 

	typedef std::vector<cv::Mat> MatVector;

	// Store the images and the points
	MatVector kinectDepthImages;
	MatVector kinectRGBImages;
	MatVector webcamRGBImages;

	bool idsLoaded = false;
	bool compared = false;
} // namespace

// Capture a frame from the webcam
cv::Mat getRGBCameraFrame()
{
    if(capture) return (IplImage *) cvQueryFrame(capture);
	else 
	{ 
		printf("webcam error"); 
		exit(0); 
	}
}

// Initialize Kinect
HRESULT initKinect() 
{
    // Get a working kinect sensor
	HRESULT hr;
    int numSensors;
    if (NuiGetSensorCount(&numSensors) < 0 || numSensors < 1) return false;
    if (NuiCreateSensorByIndex(0, &sensor) < 0) return false;

    // Initialize sensor
    sensor->NuiInitialize(NUI_INITIALIZE_FLAG_USES_DEPTH | NUI_INITIALIZE_FLAG_USES_COLOR);
	
	// Set the camera as a Depth Camera
	hr = sensor-> NuiImageStreamOpen(
		NUI_IMAGE_TYPE_DEPTH, // Depth camera
		NUI_IMAGE_RESOLUTION_640x480, // Image resolution
		0, // Image stream flags, e.g. near mode
		2, // Number of frames to buffer
		NULL, // Event handle
		&depthStream);

	if (hr != S_OK) return hr;

	// Set the camera as a RGB Camera
	hr = sensor-> NuiImageStreamOpen(
		NUI_IMAGE_TYPE_COLOR, // Depth camera
		NUI_IMAGE_RESOLUTION_640x480, // Image resolution
		0, // Image stream flags, e.g. near mode
		2, // Number of frames to buffer
		NULL, // Event handle
		&rgbStream);

    return hr;
}

// Get Kinect packed depth data 
void getKinectPackedDepthData(USHORT * dest) 
{
    NUI_IMAGE_FRAME imageFrame;
    NUI_LOCKED_RECT LockedRect;
    if (sensor->NuiImageStreamGetNextFrame(depthStream, 0, &imageFrame) < 0) return;
    INuiFrameTexture *texture = imageFrame.pFrameTexture;
    texture->LockRect(0, &LockedRect, NULL, 0);
    
	if (LockedRect.Pitch != 0)
    {
		memcpy(dest, LockedRect.pBits, width*height*sizeof(USHORT));
    }

    texture->UnlockRect(0);
    sensor->NuiImageStreamReleaseFrame(depthStream, &imageFrame);
}

// Get Kinect unpacked depth data
cv::Mat getDepthImageFromPackedData(USHORT * data)
{
	cv::Mat result(height, width, CV_16SC1);
	USHORT * data2 = result.ptr<USHORT>();
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			// Populate the matrix with actual depth in mm
			data2[y * width + x] = NuiDepthPixelToDepth(data[y * width + x]);
		}
	}

	cv::Mat result_float;
	result.convertTo(result_float, CV_32FC1);
	
	// result is CV_32FC1
	return result_float;
}

// Get RGB Image from the Kinect
void getKinectRGBData(BYTE * dest) 
{   
    NUI_LOCKED_RECT LockedRect;
    if (sensor->NuiImageStreamGetNextFrame(rgbStream, 0, &imageRGBFrame) < 0) return;
    INuiFrameTexture* texture = imageRGBFrame.pFrameTexture;
    texture->LockRect(0, &LockedRect, NULL, 0);
    if (LockedRect.Pitch != 0)
    {
		memcpy(dest, (BYTE *)LockedRect.pBits, width*height*4*sizeof(BYTE));
    }
    texture->UnlockRect(0);
    sensor->NuiImageStreamReleaseFrame(rgbStream, &imageRGBFrame);
}


// Return the depth in meters
float getDepthInMeters(USHORT * data, int x, int y)
{
	return NuiDepthPixelToDepth(data[y * width + x])/1000.0;
}

// Return the packed depth of a point
USHORT getPackedDepth(USHORT * data, int x, int y)
{
	return data[y * width + x];
}

void templateMatchingPreprocessing()
{
	FileStorage f;
	cv::Mat templIn;

	// READ - RGB template CLOSE
	if (!f.isOpened())
	{
		f.open("rgb_close.xml", FileStorage::READ);
		f["rgb_close"] >> templIn;
		f.release();
	}

	cv::Rect templRect1(243, 267, 44, 44);
	cv::Mat(templIn, templRect1).copyTo(rgb_template_1);

	cv::Rect templRect11(247, 271, 36, 36);
	cv::Mat(templIn, templRect11).copyTo(rgb_template_2);

	// READ - Kinect RGB template CLOSE
	if (!f.isOpened())
	{
		f.open("kinect_rgb_close.xml", FileStorage::READ);
		f["kinect_rgb_close"] >> templIn;
		f.release();
	}
	
	cv::Rect templRect2(236, 198, 42, 42);
	cv::Mat(templIn, templRect2).copyTo(kinect_rgb_template_1);

	cv::Rect templRect22(240, 202, 34, 34);
	cv::Mat(templIn, templRect22).copyTo(kinect_rgb_template_2);

	// READ - Kinect depth template CLOSE
	if (!f.isOpened())
	{
		f.open("kinect_depth_close.xml", FileStorage::READ);
		f["kinect_depth_close"] >> templIn;
		f.release();
	}

	cv::Rect templRect3(230, 202, 42, 42);
	cv::Mat(templIn, templRect3).copyTo(kinect_depth_template_1);

	// Kinect depth template 2 - try
	if (!f.isOpened())
	{
		f.open("kdt2.xml", FileStorage::READ);
		f["kdt2"] >> templIn;
		f.release();
	}

	cv::Rect templRect4(198, 180, 60, 60);
	//cv::Rect templRect4(208, 190, 40, 40);
	cv::Mat(templIn, templRect4).copyTo(kinect_depth_template_2);

}

// Perform the gaussian blur difference on the rgb image (initially, the rgb image is CV_8UC3 - when grabbed from the webcam)
cv::Mat getRGB_GaussianBlurDifference_32F(cv::Mat rgbImage)
{
	cv::Mat grayImage(height, width, CV_32FC1);
	cv::cvtColor(rgbImage, grayImage, CV_RGB2GRAY);
	
	cv::Mat a, b, c, d, result;
	grayImage.convertTo(a, CV_32FC1);

	cv::GaussianBlur(a, b, cv::Size(21,21), 0, 0, 4);
	cv::GaussianBlur(b, c, cv::Size(21,21), 0, 0, 4);
	cv::GaussianBlur(c, d, cv::Size(21,21), 0, 0, 4);
	result = d - c;

	// Result is returned as CV_32FC1
	return result;
}

cv::Mat getRGB_GaussianBlurDifference_32F(cv::Mat rgbImage, cv::Size s)
{
	cv::Mat grayImage(height, width, CV_32FC1);
	cv::cvtColor(rgbImage, grayImage, CV_RGB2GRAY);
	
	cv::Mat a, b, c, d, result;
	grayImage.convertTo(a, CV_32FC1);

	cv::GaussianBlur(a, b, s, 0, 0, 4);
	cv::GaussianBlur(b, c, s, 0, 0, 4);
	cv::GaussianBlur(c, d, s, 0, 0, 4);
	result = d - c;

	// Result is returned as CV_32FC1
	return result;
}

// Convert a 32F image to 8UC1 - easier to view with imshow
cv::Mat convertToDisplay(cv::Mat inputImage)
{
	double min,max;
	cv::Mat inputImageConv, inputImageConv3;
	
	cv::minMaxLoc(inputImage, &min, &max);
	inputImage.convertTo(inputImageConv, CV_8UC1, 255.0/max);

	if (inputImageConv.channels() == 1)
	{
		cv::cvtColor(inputImageConv, inputImageConv3, CV_GRAY2RGB, 3);
		return inputImageConv3;
	}

	return inputImageConv;
}

cv::Mat rescaleImage(cv::Mat inputImage)
{
	cv::Mat toShow;
	if (inputImage.cols == 640)
	{
		toShow = inputImage.clone();
	}
	else
	{
		if (inputImage.cols < 640)
		{
			// pyrUp
			cv::pyrUp(inputImage, toShow);
		}
		else
		{
			if (inputImage.cols > 640)
			{
				// pyrDown
				cv::pyrDown(inputImage, toShow);
			}	
		}
	}

	return toShow;
}

bool isInside(cv::Point toTest, cv::Point upperLeft, cv::Size s)
{
	if (toTest.x > upperLeft.x - s.width/2 && toTest.x < toTest.x + s.width/2 && toTest.y > upperLeft.y - s.height/2 && toTest.y < upperLeft.y + s.height/2)
		return true;

	return false;
}

// Depth template matching - performed directly on the depth image
bool depthTemplateMatching_32F(cv::Mat depthImage, cv::Mat depthTempl, double threshold, cv::Point * depthMatchPoint, cv::Scalar rectColor, string window_name, cv::Point median, cv::Size s)
{
	double min, max;

	// Check that input image and template have the same type and depth
	assert(depthTempl.type() == depthImage.type() || depthTempl.depth() == depthImage.depth());

	int result_cols = depthImage.cols - depthTempl.cols + 1;
	int result_rows = depthImage.cols - depthTempl.rows + 1;
	cv::Mat result;
	result.create(result_rows, result_cols, CV_32FC1);

	// Match template only accepts images of type 8U or 32F
	matchTemplate(depthImage, depthTempl, result, CV_TM_CCOEFF_NORMED);

	// Since we are using CV_TM_COEFF_NORMED -> max value is the one we are looking for
	cv::Point minLoc, maxLoc, matchLoc;
	minMaxLoc(result, &min, &max, &minLoc, &maxLoc, Mat()); 
	matchLoc = maxLoc;

	// Draw a rectangle where the matching is
	cv::Mat depth_conv = convertToDisplay(depthImage);

	depthMatchPoint->x = matchLoc.x + depthTempl.cols/2;
	depthMatchPoint->y = matchLoc.y + depthTempl.rows/2;

	line(depth_conv, Point(median.x - 20, median.y), Point(median.x + 20, median.y), Scalar(0, 255, 0), 1, 8, 0);
	line(depth_conv, Point(median.x, median.y - 20), Point(median.x, median.y + 20), Scalar(0, 255, 0), 1, 8, 0);

	cv::Mat toShow;
	if (isInside((*depthMatchPoint), median, s))
	{
		if (max < threshold)
			rectangle( depth_conv, matchLoc, Point( matchLoc.x + depthTempl.cols , matchLoc.y + depthTempl.rows ), Scalar(0,0,255), 2, 8, 0 );
		else
			rectangle( depth_conv, matchLoc, Point( matchLoc.x + depthTempl.cols , matchLoc.y + depthTempl.rows ), rectColor, 2, 8, 0 );

		toShow = rescaleImage(depth_conv);
		imshow(window_name, toShow);

		// Scale the point to the resolution 640/480
		depthMatchPoint->x *= (float) 640.0/depthImage.cols; 
		depthMatchPoint->y *= (float) 480.0/depthImage.rows;

		if (max < threshold)	return false;
	
		//printf("%s : %f \n", window_name.c_str(), max);

		return true;
	}

	toShow = rescaleImage(depth_conv);
	imshow(window_name, toShow);

	return false;
}

// Perform template matching on the rgb difference - 32F
bool rgbTemplateMatching_32F(cv::Mat rgbDifImage, cv::Mat rgbTempl, double threshold, Point * rgbMatchPoint, string window_name, cv::Point median, cv::Size s)
{
	double min, max;

	// Check that the images for template matching have the same type and depth
	assert(rgbTempl.type() == rgbDifImage.type() || rgbTempl.depth() == rgbDifImage.depth());

	int result_cols = rgbDifImage.cols - rgbTempl.cols + 1;
	int result_rows = rgbDifImage.cols - rgbTempl.rows + 1;
	cv::Mat result(result_rows, result_cols, CV_32FC1);

	// match template only accepts images of type 8U or 32F
	matchTemplate(rgbDifImage, rgbTempl, result, CV_TM_CCOEFF_NORMED);

	// since we are using CV_TM_COEFF_NORMED -> max value is the one we are looking for
	Point minLoc, maxLoc, matchLoc;
	minMaxLoc(result, &min, &max, &minLoc, &maxLoc, Mat()); 
	matchLoc = maxLoc;

	cv::Mat match_conv = convertToDisplay(rgbDifImage);

	rgbMatchPoint->x = matchLoc.x + rgbTempl.cols/2;
	rgbMatchPoint->y = matchLoc.y + rgbTempl.rows/2;

	line(match_conv, Point(median.x - 20, median.y), Point(median.x + 20, median.y), Scalar(0, 255, 0), 1, 8, 0);
	line(match_conv, Point(median.x, median.y - 20), Point(median.x, median.y + 20), Scalar(0, 255, 0), 1, 8, 0);

	cv::Mat toShow;
	if (isInside((*rgbMatchPoint), median, s))
	{
		if (max < threshold)	
			rectangle( match_conv, matchLoc, Point( matchLoc.x + rgbTempl.cols , matchLoc.y + rgbTempl.rows ), Scalar(0,0,255), 2, 8, 0 );
		else
			rectangle( match_conv, matchLoc, Point( matchLoc.x + rgbTempl.cols , matchLoc.y + rgbTempl.rows ), Scalar(0,255,0), 2, 8, 0 );
		
		toShow = rescaleImage(match_conv);
		imshow(window_name, toShow);

		// Scale the point to the resolution 640/480
		rgbMatchPoint->x *= (float) 640.0/rgbDifImage.cols; 
		rgbMatchPoint->y *= (float) 480.0/rgbDifImage.rows;

		if (max < threshold)	return false;

		//printf("%s : %f \n", window_name.c_str(), max);

		return true;
	}

	toShow = rescaleImage(match_conv);
	imshow(window_name, toShow);

	return false;
}

// Reproject a 3D point to a 2D point
void reproject(const double a[12], double u, double v, double z, double * r) 
{
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

// Recontruct the depth image from the color image
// white points - no depth info from Kinect
// black points - reprojection out of bounds
cv::Mat getDepthColorReconstruction(cv::Mat depthImage, cv::Mat rgbImage, USHORT * data)
{
	cv::Mat coloredDepth(depthImage.size(), CV_8UC3);
	assert(coloredDepth.channels() == 3);
	assert(rgbImage.channels() == 3);

	for (int y=0; y<coloredDepth.rows; ++y)
	{
		for (int x=0; x<coloredDepth.cols; ++x)
		{
			int x_col, y_col;
			double r[2];
			float depthInM = getDepthInMeters(data, x, y);
			
			if (depthInM != 0)
			{
				reproject(webcam_a_calib, x, y, depthInM, r);
				x_col = r[0];
				y_col = r[1];
			
				if ((x_col >= 0 && x_col < width) && (y_col >= 0 && y_col < height))
				{
					// take the color from x_col and y_col and project it in the depth image
					for (int c = 0; c < 3; ++c) 
					{
						coloredDepth.ptr<uchar>(y)[x * 3 + c] = rgbImage.ptr<uchar>(y_col)[x_col * 3 + c]; 
					}
				}
				else
				{
					for (int c = 0; c < 3; ++c) 
					{
						coloredDepth.ptr<uchar>(y)[x * 3 + c] = 0; 
					}
				}
			}
			else
			{
				// We have no depth information about this points (depth from kinect = 0)
				for (int c = 0; c < 3; ++c) 
				{
					coloredDepth.ptr<uchar>(y)[x * 3 + c] = 0; 
				}
			}
		}
	}

	return coloredDepth;
}

// Distance between two points in pixels
float getDistance(Point2f p1, Point2f p2)
{
	return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y));
}

// Copy the result of a new calibration to webcam_a_calib - global var used for the reprojection
void setReprojectionMatrix(cv::Mat calibMatrix, double * calib)
{
	for (unsigned i = 0; i<12; ++i)
	{
		calib[i] = calibMatrix.at<double>(i, 0);
	}
}

// Get reprjected points
Point2DVector getReprojectedPoints(Point3DVector points3D, double * a)
{
	Point2DVector reprojections;
	reprojections.clear();
	for (unsigned i=0; i<points3D.size(); ++i)
	{
		double r[2];
		reproject(a, points3D[i].x, points3D[i].y, points3D[i].z, r);
		int x = r[0];
		int y = r[1];
		reprojections.push_back(cv::Point2f(x, y));
	}

	return reprojections;
}

// Remove the points that are outliers
void removeOutliers(Point2DVector * projections, Point3DVector * points3D, Point2DVector reprojections, float error, vector<int> pointIds, Point2DVector * newProjections, Point3DVector * newPoints3D, vector<int> * newPointIds)
{
	newProjections->clear();
	newPoints3D->clear();
	newPointIds->clear();

	for (unsigned i=0; i<projections->size(); ++i)
	{
		Point2f pProj = projections->at(i);
		Point2f pReproj = reprojections.at(i);
		float dist = getDistance(pProj, pReproj);
		if (dist < error)
		{
			// add points from projections and from points3D
			newProjections->push_back(projections->at(i));
			newPoints3D->push_back(points3D->at(i));
			newPointIds->push_back(pointIds.at(i));
		}
	}
}

// Create a new vector of ids
vector<int> initPointIds(int size, bool loaded)
{
	vector<int> ids;

	if (!loaded)
	{
		// initialize ids to i
		for (int i=0; i<size; ++i)
		{
			int val = i;
			ids.push_back(val);
		}

		return ids;
	}
	else
	{
		// read from file
		std::ifstream f;
		f.open("compare\\ids.txt");
		if (f.is_open())
		{
			while (!f.eof())
			{
				int val;
				f >> val;
				ids.push_back(val);
			}
		}

		f.close();

		return ids;
	}
}

void saveIds(vector<int> ids)
{
	std::ofstream f;
	f.open ("compare\\ids.txt");
	for (unsigned i = 0; i<ids.size(); ++i)
	{
		f << ids.at(i) << "\n";
	}

	f.close();
}


// Iterative improvement of calibration by eliminating outliers
vector<int> iterativeImprovementCalibration(Calibrator * c, cv::Mat * calibResult, double * a, unsigned * minCalibPoints, string fileName, bool * calibrated, vector<int> ids)
{
	if (c->numEntries() == *minCalibPoints)
	{
		float error = 10000;
		while (error > 2)
		{
			// Webcam calibration
			*calibResult = c->calibrate();
			setReprojectionMatrix(*calibResult, a);

			Point2DVector projections = c->projections();
			Point3DVector points3D = c->points3D();
			Point2DVector reprojections = getReprojectedPoints(points3D, a);

			Point2DVector newProjections;
			Point3DVector newPoints3D;
			vector<int> newIds;
			removeOutliers(&projections, &points3D, reprojections, error, ids, &newProjections, &newPoints3D, &newIds); 

			ids = newIds;
			c->setProjections(newProjections);
			c->setPoints3D(newPoints3D);
			*minCalibPoints = newProjections.size();

			*calibResult = c->calibrate();
			setReprojectionMatrix(*calibResult, a);
			
			error /= 2;
		}
		
		c->save(fileName);
		*calibrated = true;
	}

	return ids;
}


// Provide a debug way for the projections
cv::Mat debugProjections(const cv::Mat rgbImage)
{
	cv::Mat rClone = rgbImage.clone();

	Point2DVector projections = webcamCalibrator.projections();
	Point3DVector points3D = webcamCalibrator.points3D();

	for (unsigned i = 0; i < webcamCalibrator.numEntries(); ++i) 
	{
		char * s; 
		s = (char *) malloc (10 * sizeof(char));
		sprintf_s(s, sizeof(s), "%d", i);
		putText(rClone, s, Point(projections[i].x, projections[i].y), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0,255,0), 1, CV_AA);
		free(s);
	}

	if (webcamCalibrator.numEntries() >= webcamMinCalibrationPoints)
	{
		for (unsigned i = 0; i < webcamCalibrator.numEntries(); ++i) 
		{
			// Draw reprojection with red
			double r[2];
			reproject(webcam_a_calib, points3D[i].x, points3D[i].y, points3D[i].z, r);
			int x_col = r[0];
			int y_col = r[1];

			// Draw Connection between the points
			line(rClone, Point(x_col, y_col), Point(projections[i].x, projections[i].y), Scalar(255, 0, 0), 1, 8, 0);

			char * s; 
			s = (char *) malloc (10 * sizeof(char));
			sprintf_s(s, sizeof(s), "%d", i);
			putText(rClone, s, Point(x_col, y_col), FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0,0,255), 1, CV_AA);
			free(s);
		}
	}

	return rClone;
}

// Make sure that two points are far away from each other
bool checkPoints(cv::Point a, cv::Point b, int threshold)
{
	float dist = sqrt((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));

	if (dist < threshold)	
		return false;

	return true;
}

// Filter green color, tip of the pattern
cv::Mat getColorMask(cv::Mat inputImg, int w, int h, string window_name, cv::Scalar min_color, cv::Point * median) // width and height of a window
{
	cv::Mat colorFilter, imgHsv, imgRes;
	cv::cvtColor(inputImg, imgHsv, CV_RGB2HSV, 3);//convert the color space

	cv::Scalar max_color(75,255,255);
	cv::inRange(imgHsv, min_color,max_color, colorFilter);//search for the color in image

	//imshow(window_name, colorFilter);
	imgRes = colorFilter.clone();

	// Find median X and median Y
	int ww = inputImg.size().width;
	int hh = inputImg.size().height;
	int xArr[640];
	for (int i=0; i<640; ++i)
		xArr[i] = 0;
	int yArr[480];
	for (int i=0; i<480; ++i)
		yArr[i] = 0;

	int countX = 0, countY = 0;
	for (int y=0; y<hh; ++y)
	{
		for (int x = 0; x<ww; ++x)
		{
			if (imgRes.ptr<uchar>(y)[x] > 0)
			{
				if (xArr[x] == 0)
				{
					xArr[x] = 1;
					countX++;
				}
				
				if (yArr[y] == 0)
				{
					yArr[y] = 1;
					countY++;
				}
			}
		}
	}

	int medX = countX/2;
	int medY = countY/2;

	int i = 0;
	while (medX > 0)
	{
		if (xArr[i] == 1)
			medX--;
		i++;
	}
	int cX = i;

	i = 0;
	while (medY > 0)
	{
		if (yArr[i] == 1)
			medY--;
		i++;
	}
	int cY = i;

	// Draw the window if there are some points in the image
	if (i != 0)
	{
		for (int y=0; y<hh; ++y)
		{
			for (int x=0; x<ww; ++x)
			{
				if ((x >= 0 && x < inputImg.size().width) && (y >= 0 && y < inputImg.size().height))
					imgRes.ptr<uchar>(y)[x] = 0; 
			}
		}

		for (int y=cY-h/2; y<=cY+h/2; ++y)
		{
			for (int x=cX-w/2; x<=cX+w/2; ++x)
			{
				if ((x >= 0 && x < inputImg.size().width) && (y >= 0 && y < inputImg.size().height))
					imgRes.ptr<uchar>(y)[x] = 255; 
			}
		}
	}

	median->x = cX;
	median->y = cY;
	
	//imshow(window_name, imgRes);

	return imgRes;
}

// Filter an image based on the color mask that is provided as input 
cv::Mat filterImageByColor(cv::Mat inputImg, cv::Mat maskImg)
{
	cv::Mat resultImg = inputImg.clone();
	int w = inputImg.size().width;
	int h = inputImg.size().height;

	for (int y=0; y<h; ++y)
	{
		for (int x=0; x<w; ++x)
		{
			if (maskImg.ptr<uchar>(y)[x] > 0)
			{
				// copy image
				for (int c = 0; c < inputImg.channels(); ++c) 
				{
					resultImg.ptr<uchar>(y)[x * inputImg.channels() + c] = inputImg.ptr<uchar>(y)[x * inputImg.channels() + c]; 
				}
			}
			else
			{
				// black
				for (int c = 0; c < inputImg.channels(); ++c) 
				{
					resultImg.ptr<uchar>(y)[x * inputImg.channels() + c] = 0; 
				}
			}
		}
	}

	return resultImg;
}

// Filter depth image by the color mask
cv::Mat filterDepthImageByColor(cv::Mat inputImg, cv::Mat maskImg, USHORT * data)
{
	cv::Mat resultImg = inputImg.clone();
	int w = inputImg.size().width;
	int h = inputImg.size().height;

	for (int y=0; y<h; ++y)
	{
		for (int x=0; x<w; ++x)
		{
			// copy image
			USHORT pixelDepth = data[y*width+x];
			if (pixelDepth != 0)
			{
				long xx, yy;
				NuiImageGetColorPixelCoordinatesFromDepthPixelAtResolution(NUI_IMAGE_RESOLUTION_640x480, NUI_IMAGE_RESOLUTION_640x480, &imageRGBFrame.ViewArea, x, y, pixelDepth, &xx, &yy);

				if ((xx >= 0 && xx < maskImg.size().width) && (yy >= 0 && yy < maskImg.size().height))
				{
					if (maskImg.ptr<uchar>(yy)[xx] > 0)
						resultImg.ptr<float>(y)[x] = inputImg.ptr<float>(y)[x];
					else
						resultImg.ptr<float>(y)[x] = 0;
				}
				else
				{
					resultImg.ptr<float>(y)[x] = 0; 
				}
			}
		}
	}

	return resultImg;
}

// Filter an image and get only the resulting window
cv::Mat filterByMedian(cv::Mat inputImg, cv::Point median, cv::Size s)
{
	cv::Mat result;

	if (median.x - s.width/2 > 0 && median.y - s.height/2 > 0)
	{
		cv::Rect templRect(median.x - s.width/2, median.y - s.height/2, s.width, s.height);
		cv::Mat(inputImg, templRect).copyTo(result);
	}
	else
		return inputImg;

	return result;
}

// Get Kinect depth point from RGB image of the Kinect 
bool getDepthPointFromRGB(cv::Mat depthImg, USHORT * data, cv::Point rgbPoint, cv::Point * newPoint)
{
	int w = depthImg.size().width;
	int h = depthImg.size().height;

	for (int y=0; y<h; ++y)
	{
		for (int x=0; x<w; ++x)
		{
			// copy image
			USHORT pixelDepth = data[y*width+x];
			if (pixelDepth != 0)
			{
				long xx, yy;
				NuiImageGetColorPixelCoordinatesFromDepthPixelAtResolution(NUI_IMAGE_RESOLUTION_640x480, NUI_IMAGE_RESOLUTION_640x480, &imageRGBFrame.ViewArea, x, y, pixelDepth, &xx, &yy);

				if (xx == rgbPoint.x && yy == rgbPoint.y)
				{
					newPoint->x = xx;
					newPoint->y = yy;
					return true;
				}
			}
		}
	}

	return false;
}

// Get the corresponding RGB Point based on the depth point
bool getRGBPointFromDepth(cv::Mat rgbImg, cv::Point depthPoint, USHORT * data, cv::Point * newPoint)
{
	long x, y;
	USHORT pixelDepth = getPackedDepth(data, depthPoint.x, depthPoint.y);
	NuiImageGetColorPixelCoordinatesFromDepthPixelAtResolution(NUI_IMAGE_RESOLUTION_640x480, NUI_IMAGE_RESOLUTION_640x480, &imageRGBFrame.ViewArea, depthPoint.x, depthPoint.y, pixelDepth, &x, &y);

	if ((x > 0 && x < rgbImg.size().width) && (y > 0 && y < rgbImg.size().height))
	{
		newPoint->x = x;
		newPoint->y = y;
		return true;
	}

	return false;
}

// Write an image to a file and place an X where the "point" indicates
void writeToFile(cv::Mat inputImg, cv::Point point, string path)
{
	cv::Mat res = inputImg.clone();

	line(res, Point(point.x - 20, point.y), Point(point.x + 20, point.y), Scalar(0, 255, 0), 2, 8, 0);
	line(res, Point(point.x, point.y - 20), Point(point.x, point.y + 20), Scalar(0, 255, 0), 2, 8, 0);

	imwrite(path, convertToDisplay(res));
}

// Write an image to a file and place an X where the detected position is and another X where it should be
void writeToFile(cv::Mat inputImg, cv::Point point1, cv::Point point2, string path)
{
	cv::Mat res = convertToDisplay(inputImg);

	// Green for first point
	line(res, Point(point1.x - 20, point1.y), Point(point1.x + 20, point1.y), Scalar(0, 255, 0), 2, 8, 0);
	line(res, Point(point1.x, point1.y - 20), Point(point1.x, point1.y + 20), Scalar(0, 255, 0), 2, 8, 0);

	// Red is for secont point
	line(res, Point(point2.x - 20, point2.y), Point(point2.x + 20, point2.y), Scalar(0, 0, 255), 2, 8, 0);
	line(res, Point(point2.x, point2.y - 20), Point(point2.x, point2.y + 20), Scalar(0, 0, 255), 2, 8, 0);

	imwrite(path, res);
}

void compareKinectCalibration(USHORT * data, vector<int> ids)
{
	int size = kinectDepthImages.size();
	Point2DVector projections = kinectCalibrator.projections();
	Point3DVector points3D = kinectCalibrator.points3D();
	Point2DVector reprojections = getReprojectedPoints(points3D, kinect_a_calib);

	std::ofstream f;
	f.open ("compare_output\\error.txt");
	f << "Our Calibration" << "\t" << "Kinect's Calibration" << "\n";
	for (unsigned i = 0; i<points3D.size(); i++)
	{
		cv::Point3f point3D(points3D.at(i).x, points3D.at(i).y, getDepthInMeters(data,points3D.at(i).x,points3D.at(i).y));

		int x1 = reprojections.at(i).x;
		int y1 = reprojections.at(i).y;

		long x2, y2;
		USHORT pixelDepth = getPackedDepth(data, point3D.x, point3D.y);
		NuiImageGetColorPixelCoordinatesFromDepthPixelAtResolution(NUI_IMAGE_RESOLUTION_640x480, NUI_IMAGE_RESOLUTION_640x480, &imageRGBFrame.ViewArea, point3D.x, point3D.y, pixelDepth, &x2, &y2);
		
		int imgId = ids.at(i);
		cv::Mat depthImage = kinectDepthImages.at(imgId);
		cv::Mat kinectImage = kinectRGBImages.at(imgId);

		// Cross on depth image
		line(depthImage, Point(point3D.x - 20, point3D.y), Point(point3D.x + 20, point3D.y), Scalar(0, 255, 0), 2, 8, 0);
		line(depthImage, Point(point3D.x, point3D.y - 20), Point(point3D.x, point3D.y + 20), Scalar(0, 255, 0), 2, 8, 0);

		int x3 = projections.at(i).x;
		int y3 = projections.at(i).y;
		// Blue cross on template matching
		line(kinectImage, Point(x3 - 20, y3), Point(x3 + 20, y3), Scalar(255, 0, 0), 2, 8, 0);
		line(kinectImage, Point(x3, y3 - 20), Point(x3, y3 + 20), Scalar(255, 0, 0), 2, 8, 0);

		char path[1024];
			
		sprintf_s(path, 1024, "compare_output\\original_depth_%d.jpg", imgId);
		writeToFile(depthImage, cv::Point(point3D.x, point3D.y), path); 

		sprintf_s(path, 1024, "compare_output\\rgb_image_%d.jpg", imgId);
		writeToFile(kinectImage, cv::Point(x1,y1), cv::Point(x2,y2), path); // Green detected, red computed from Kinect

		// compute the distance between P1-P3 and P2-P3
		cv::Point2f P1(x1, y1);
		cv::Point2f P2(x2, y2);
		cv::Point2f P3(x3, y3);
		float d1 = getDistance(P1, P3);
		float d2 = getDistance(P2, P3);

		f << d1 << "\t" << d2 << "\n";
	}

	f.close();
}

// Save the impages for the compare in a file
void saveCompareImages()
{
	char path[1024];

	for (unsigned i=0; i<kinectDepthImages.size(); i++)
	{
		sprintf_s(path, 1024, "compare\\original_depth_%d.jpg", i);
		imwrite(path, convertToDisplay(kinectDepthImages.at(i)));
	}

	for (unsigned i=0; i<kinectRGBImages.size(); i++)
	{
		sprintf_s(path, 1024, "compare\\rgb_image_%d.jpg", i);
		imwrite(path, convertToDisplay(kinectRGBImages.at(i)));
	}
}

// Load the images for the compare from a file
void loadCompareImages(int size)
{
	char path[1024];

	kinectDepthImages.clear();
	kinectRGBImages.clear();

	for (unsigned i=0; i<size; i++)
	{
		sprintf_s(path, 1024, "compare\\original_depth_%d.jpg", i);
		cv::Mat img = imread(path);
		kinectDepthImages.push_back(img);

		sprintf_s(path, 1024, "compare\\rgb_image_%d.jpg", i);
		cv::Mat img2 = imread(path);
		kinectRGBImages.push_back(img2);
	}
}

void draw(cv::Mat depthImage, USHORT * data, cv::Mat rgbImage);

void loop()
{
	USHORT data[width*height];// array containing the depth information of each pixel
	getKinectPackedDepthData(data);

	// Kinect: Get depth data
	cv::Mat original_depth = getDepthImageFromPackedData(data);

	// Kinect: Get RGB data
	getKinectRGBData(rgbData);
	cv::Mat kinectRGBImage(height, width, CV_8UC4, rgbData);
	//imshow("Kinect orig image", kinectRGBImage);

	// Get RGB data
	cv::Mat image = getRGBCameraFrame();
	//imshow("Original image", image);
	cv::Mat rgbImage;
	cv::flip(image, rgbImage, 1);

	cv::Mat debugImg = debugProjections(rgbImage);
	imshow("Webcam: RGB Image", debugImg);

	// Calibrate Webcam
	if (webcamCalibrator.numEntries() == webcamMinCalibrationPoints && !webcamCalibrated)
	{
		vector<int> ids = initPointIds(webcamMinCalibrationPoints, false);
		iterativeImprovementCalibration(&webcamCalibrator, &webcam_calib_result, webcam_a_calib, &webcamMinCalibrationPoints, "webcam_clicks.yml", &webcamCalibrated, ids);

		// save images
		saveCompareImages();
	}

	// Calibrate Kinect
	vector<int> kinectIds; 	
	if (kinectCalibrator.numEntries() == kinectMinCalibrationPoints && !kinectCalibrated)
	{
		vector<int> ids = initPointIds(kinectMinCalibrationPoints, idsLoaded);
		kinectIds = iterativeImprovementCalibration(&kinectCalibrator, &kinect_calib_result, kinect_a_calib, &kinectMinCalibrationPoints, "kinect_clicks.yml", &kinectCalibrated, ids);
		saveIds(kinectIds);
	}

	// Depth color reconstruction
	if (webcamCalibrated == true)
	{
		cv::Mat depthReconstr = getDepthColorReconstruction(original_depth, rgbImage, data);
		cv::Mat depthReconst_flipped;
		cv::flip(depthReconstr, depthReconst_flipped, 1);
		imshow("depth reconstr", depthReconst_flipped);
		//imshow("original depth", convertToDisplay(original_depth));

		imwrite("report_img\\depth_reconstr.jpg", depthReconst_flipped);

		draw(original_depth, data, rgbImage);
	}

	if (kinectCalibrated == true && compared == false)
	{
		compareKinectCalibration(data, kinectIds);
		compared = true;
	}

	// Template matching algorithm
	if (frameCount % 13 == 0 && !webcamCalibrated)
	{
		Point medianWebcam, medianKinect;
		cv::Size s(100,100);

		// RGB filtering on webcam
		cv::Scalar min_color(38,100,100);
		cv::Mat mask = getColorMask(rgbImage, 100, 100, "webcam filtering", min_color, &medianWebcam);
		cv::Mat rgbFilteredImage = filterImageByColor(rgbImage, mask);
	
		// RGB filterig on kinect
		cv::Scalar min_color2(38,110,110);
		mask = getColorMask(kinectRGBImage, 100, 100, "kinect rgb filtering", min_color2, &medianKinect);
		cv::Mat kinectRGBFilteredImage = filterImageByColor(kinectRGBImage, mask);

		// Depth filtering on the Kinect
		cv::Mat kinectDepthFilteredImage = filterDepthImageByColor(original_depth, mask, data);
		cv::Point dMedian; 
		getDepthPointFromRGB(original_depth, data, medianKinect, &dMedian);

		cv::Mat rgbDif = getRGB_GaussianBlurDifference_32F(rgbFilteredImage);
		cv::Mat kinectRGBDif = getRGB_GaussianBlurDifference_32F(kinectRGBFilteredImage);

		cv::Mat rgbDif_2 = getRGB_GaussianBlurDifference_32F(rgbFilteredImage, cv::Size(35,35));
		cv::Mat kinectRGBDif_2 = getRGB_GaussianBlurDifference_32F(kinectRGBFilteredImage, cv::Size(35,35));
	
		cv::Mat rgbDif_red, kinectRGBDif_red;
		cv::pyrDown(rgbDif_2, rgbDif_red);
		cv::pyrDown(kinectRGBDif_2, kinectRGBDif_red);

		cv::Mat depthImg_double;
		cv::pyrUp(kinectDepthFilteredImage, depthImg_double);

		bool kinectRGBRes, kinectDepthRes, rgbRes;
		Point kinectRGBMatchingPoint, depthMatchingPoint, rgbMatchingPoint, computedDepthMatchingPoint, computedKinectRGBMatchingPoint_1,computedKinectRGBMatchingPoint_2;

		// Try to find the image in the close region
		kinectRGBRes = rgbTemplateMatching_32F(kinectRGBDif_red, kinect_rgb_template_2, 0.75, &kinectRGBMatchingPoint, "kinect_rgb_matching", cv::Point(medianKinect.x/2, medianKinect.y/2), cv::Size(s.width/2, s.height/2));		
		kinectDepthRes = depthTemplateMatching_32F(kinectDepthFilteredImage, kinect_depth_template_2, 0.8, &depthMatchingPoint, Scalar(0,255,0), "depth_matching", dMedian, s);

		if (kinectRGBRes == true && kinectDepthRes == true)
		{
			rgbRes = rgbTemplateMatching_32F(rgbDif_red, rgb_template_2, 0.75, &rgbMatchingPoint, "rgb_matching", cv::Point(medianWebcam.x/2, medianWebcam.y/2), cv::Size(s.width/2, s.height/2));
		}
		else
		{
			kinectDepthRes = depthTemplateMatching_32F(depthImg_double, kinect_depth_template_2, 0.82, &depthMatchingPoint, Scalar(0,255,0), "depth_matching", cv::Point(dMedian.x*2, dMedian.y*2), cv::Size(s.width*2, s.height*2));
			kinectRGBRes = rgbTemplateMatching_32F(kinectRGBDif, kinect_rgb_template_1, 0.75, &kinectRGBMatchingPoint, "kinect_rgb_matching", medianKinect, s);	
			rgbRes = rgbTemplateMatching_32F(rgbDif, rgb_template_1, 0.75, &rgbMatchingPoint, "rgb_matching", medianWebcam, s);
		}

		if (kinectRGBRes == true && kinectDepthRes == true && rgbRes == true)
		{
			bool getP = getDepthPointFromRGB(original_depth, data, kinectRGBMatchingPoint, &computedDepthMatchingPoint); // kinect rgb template result -> kinect depth (using calibration)
			bool getP_rgb1 = getRGBPointFromDepth(kinectRGBImage, depthMatchingPoint, data, &computedKinectRGBMatchingPoint_1); // kinect depth template result -> kinect rgb (using calibration)
			bool getP_rgb2 = getRGBPointFromDepth(kinectRGBImage, computedDepthMatchingPoint, data, &computedKinectRGBMatchingPoint_2);

			cv::Point depthChoice;
			float d1 = getDistance(kinectRGBMatchingPoint, computedKinectRGBMatchingPoint_1);
			float d2 = getDistance(kinectRGBMatchingPoint, computedKinectRGBMatchingPoint_2);

			if (getP && getP_rgb1 && (d1 < 8))
			{
				// all conditions have been met, add points to the webcam calibrator
				webcamCalibrator.add3DPoint(depthMatchingPoint.x, depthMatchingPoint.y, getDepthInMeters(data, depthMatchingPoint.x, depthMatchingPoint.y));
				webcamCalibrator.addProjCam(rgbMatchingPoint.x, rgbMatchingPoint.y);

				// add points to the kinect calibrator
				kinectCalibrator.add3DPoint(depthMatchingPoint.x, depthMatchingPoint.y, getDepthInMeters(data, depthMatchingPoint.x, depthMatchingPoint.y));
				kinectCalibrator.addProjCam(kinectRGBMatchingPoint.x, kinectRGBMatchingPoint.y);

				// images that need to be saved: original_depth, rgbImage and kinectRGBImage
				// need to store the images

				// Debug images
				int no = webcamCalibrator.numEntries();

				if (no == 5)
				{
					imwrite("report_img\\original_depth.jpg", convertToDisplay(original_depth));
					imwrite("report_img\\rgb_image.jpg", rgbImage);
					imwrite("report_img\\kinect_rgb_image.jpg", kinectRGBImage);
					imwrite("report_img\\color_mask.jpg", convertToDisplay(mask));
					imwrite("report_img\\rgb_dif.jpg", convertToDisplay(rgbDif));
					imwrite("report_img\\kinect_rgb_dif.jpg", convertToDisplay(kinectRGBDif));
				}
				
				char path[1024];
			
				sprintf_s(path, 1024, "debug_img\\original_depth_%d.jpg", (no-1));
				writeToFile(original_depth, depthMatchingPoint, computedDepthMatchingPoint, path); // Green detected, red computed

				sprintf_s(path, 1024, "debug_img\\rgb_image_%d.jpg", (no-1));
				writeToFile(rgbImage, rgbMatchingPoint, path);

				sprintf_s(path, 1024, "debug_img\\kinect_rgb_image_%d.jpg", (no-1));
				writeToFile(kinectRGBImage, kinectRGBMatchingPoint, computedKinectRGBMatchingPoint_1, path); // Green detected, red computed

				// Images 
				cv::Mat d;
				original_depth.copyTo(d);
				kinectDepthImages.push_back(d);

				cv::Mat k;
				kinectRGBImage.copyTo(k);
				kinectRGBImages.push_back(k);

				printf("Matching point: %d out of %d \n", webcamCalibrator.numEntries(), webcamMinCalibrationPoints);
			}
		}

		frameCount = 1;
	}
	else
	{
		frameCount++;
	}
}

// --------------------------------------------------------------------------------------------------------------------
// OpenGL 
// --------------------------------------------------------------------------------------------------------------------

//angle of rotation
float xpos = 0, ypos = 2, zpos = -8, xrot = 0, yrot = 180;

float lastx, lasty;

void camera (void) 
{
    glRotatef(xrot,1.0,0.0,0.0);  //rotate our camera on teh x-axis (left and right)
    glRotatef(yrot,0.0,1.0,0.0);  //rotate our camera on the y-axis (up and down)
	glTranslated(-xpos,-ypos,-zpos); //translate the screen to the position of our camera
}

void reshape (int w, int h)
{
    glViewport (0, 0, (GLsizei)w, (GLsizei)h); //set the viewport to the current window specifications
    glMatrixMode (GL_PROJECTION); //set the matrix to projection

    glLoadIdentity ();
    gluPerspective(60, (GLfloat)w / (GLfloat)h, 1, 20.0); //set the perspective (angle of sight, width\height, zNear, zFar)
    glMatrixMode (GL_MODELVIEW); //set the matrix back to model
}

void specialKeys( int key, int x, int y ) 
{ 
	switch (key)
	{
		case GLUT_KEY_RIGHT: 
			yrot += 1;
			if (yrot >360) 
				yrot -= 360;
			break;

		case GLUT_KEY_LEFT:
			yrot -= 1;
			if (yrot < -360) 
				yrot += 360;
			break; 

		case GLUT_KEY_DOWN: 
			xrot += 1;
			if (xrot >360) 
				xrot -= 360;
			break;

		case GLUT_KEY_UP:
			xrot -= 1;
			if (xrot < -360) 
				xrot += 360;
			break;

		default: 
			break;
	}
 
	//  Request display update
	glutPostRedisplay(); 
}

void keyboard (unsigned char key, int x, int y) 
{
	if (key=='q')
    {
		float xrotrad;
		xrotrad = (xrot / 180 * 3.141592654f);
		ypos += float(cos(xrotrad)) * 0.2;
		zpos += float(sin(xrotrad)) * 0.2;
    }

    if (key=='e')
    {
		float xrotrad;
		xrotrad = (xrot / 180 * 3.141592654f);
		ypos -= float(cos(xrotrad)) * 0.2;
		zpos -= float(sin(xrotrad)) * 0.2;
    }

    if (key=='w')
	{
		float xrotrad, yrotrad;
		yrotrad = (yrot / 180 * 3.141592654f);
		xrotrad = (xrot / 180 * 3.141592654f); 
		xpos += float(sin(yrotrad)) ;
		zpos -= float(cos(yrotrad)) ;
		ypos -= float(sin(xrotrad)) ; 
	}
	
    if (key=='s')
    {
		float xrotrad, yrotrad;
		yrotrad = (yrot / 180 * 3.141592654f);
		xrotrad = (xrot / 180 * 3.141592654f); 
		xpos -= float(sin(yrotrad));
		zpos += float(cos(yrotrad)) ;
		ypos += float(sin(xrotrad));
    }

    if (key=='d')
    {
		float yrotrad;
		yrotrad = (yrot / 180 * 3.141592654f);
		xpos += float(cos(yrotrad)) * 0.2;
		zpos += float(sin(yrotrad)) * 0.2;
    }

    if (key=='a')
    {
		float yrotrad;
		yrotrad = (yrot / 180 * 3.141592654f);
		xpos -= float(cos(yrotrad)) * 0.2;
		zpos -= float(sin(yrotrad)) * 0.2;
    }

    if (key==27)
    {
		exit(0);
    }

	printf("xpos = %f, ypos = %f, zpos = %f, xrot = %f \n", xpos, ypos, zpos, xrot);
}

bool init(int argc, char* argv[]) 
{
	// OpenGL init
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    
	glutInitWindowSize(width,height);
	glutInitWindowPosition (0, 0);
	
    glutCreateWindow("Kinect 3D Point Cloud");
	
	glutDisplayFunc(loop);
    glutIdleFunc(loop);
	glutReshapeFunc (reshape);

    glutKeyboardFunc (keyboard); 
	glutSpecialFunc(specialKeys);
	
	return true;
}

void drawKinectPointCloud(cv::Mat depthImage, USHORT * data, cv::Mat rgbImage)
{
	glDisable(GL_CULL_FACE);

	cv::Mat depth_flipped, rgbImage_flipped;

	// Display the points as a 3D point cloud
	glBegin(GL_POINTS);
		for (int y = 0; y < height; ++y)
		{
			for (int x = 0; x < width; ++x)
			{
				int x_col, y_col;
				double r[2];
				float depthInM = getDepthInMeters(data, x, y);
				float depthInMM = NuiDepthPixelToDepth(getPackedDepth(data,x,y));

				// Get 3D coordinates from Depth Image Space
				//Vector4 pointToDisplay = NuiTransformDepthImageToSkeleton(x, y, depthInMM);
				Vector4 pointToDisplay = NuiTransformDepthImageToSkeleton(x, y, getPackedDepth(data,x,y));

				if (depthInMM != 0)
				{
					reproject(webcam_a_calib,x,y,depthInM, r);
					x_col = r[0];
					y_col = r[1];

					if ((x_col >= 0 && x_col < rgbImage.cols) && (y_col >= 0 && y_col < rgbImage.rows))
					{
						// take the color from x_col and y_col and project it in the depth image
						int blue = (int) rgbImage.ptr<uchar>(y_col)[x_col * 3 + 0]; 
						int green = (int) rgbImage.ptr<uchar>(y_col)[x_col * 3 + 1]; 
						int red = (int) rgbImage.ptr<uchar>(y_col)[x_col * 3 + 2]; 
						
						// Vertex Color
						glColor3f((float)red/255.0, (float)green/255.0, (float)blue/255.0);
						glVertex3f(pointToDisplay.x, pointToDisplay.y, pointToDisplay.z);
					}
				}				
			}
		}
	glEnd();
}


void draw(cv::Mat depthImage, USHORT * data, cv::Mat rgbImage)
{
	glClearColor (0.0,0.0,0.0,1.0); //clear the screen to black
    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); //clear the color buffer and the depth buffer
    glLoadIdentity();  

    camera();
    drawKinectPointCloud(depthImage, data, rgbImage); // drawing function

    glutSwapBuffers(); //swap the buffers
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Main 
int main(int argc, char* argv[])
{
	// init opengl
	if (!init(argc, argv)) return 1;

	// init Kinect
	if (initKinect() != S_OK)	return -1;

	// init webcam
    if(!(capture = cvCaptureFromCAM(0)))	return -1;

	// Webcam calibration
	webcam_calib_result = webcamCalibrator.load("webcam_clicks.yml");
	webcamMinCalibrationPoints = webcamCalibrator.numEntries();
	webcam_calib_result = webcamCalibrator.calibrate();
	//webcamCalibrated = true;
	setReprojectionMatrix(webcam_calib_result, webcam_a_calib);

	//// Kinect calibration
	kinect_calib_result = kinectCalibrator.load("kinect_clicks.yml");
	kinectMinCalibrationPoints = kinectCalibrator.numEntries();
	kinect_calib_result = kinectCalibrator.calibrate();
	//kinectCalibrated = true;
	setReprojectionMatrix(kinect_calib_result, kinect_a_calib);
	
	loadCompareImages(30);
	idsLoaded = true;

	// Preprocessing
	templateMatchingPreprocessing();

	glutMainLoop();

	// Release the capture device and destroy windows
	cvReleaseCapture(&capture);

	return 0;
}