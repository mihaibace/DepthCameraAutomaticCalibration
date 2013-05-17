#include <Windows.h>
#include <Ole2.h>

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

	GLubyte rgbData[width*height*4];
	NUI_IMAGE_FRAME imageRGBFrame;

	// Webcam variables
	CvCapture* capture;

	// Calibrator 
	Calibrator kinectCalibrator;
	cv::Mat calibResult;
	double aCalib[12];

	// Define minimum number of calibration points
	unsigned minCalibrationPoints = 20;

	// Template matching
	cv::Mat rgbTemplate;
	cv::Mat depthTemplate;

	// Frame skipping variable
	int frameCount = 0;
	bool calibrated = false;
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
	USHORT * data2 = (USHORT *) malloc(height * width * sizeof(USHORT));
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			// Populate the matrix with actual depth in mm
			data2[y * width + x] = NuiDepthPixelToDepth(data[y * width + x]);
		}
	}

	cv::Mat result_float;

	cv::Mat result(height, width, CV_16SC1, data2);
	result.convertTo(result_float, CV_32FC1);
	
	// result is CV_32FC1
	return result_float;
}

// Get Kinect RGB data
void getKinectRGBData(GLubyte * dest) 
{   
    NUI_LOCKED_RECT LockedRect;
    if (sensor->NuiImageStreamGetNextFrame(rgbStream, 0, &imageRGBFrame) < 0) return;
    INuiFrameTexture* texture = imageRGBFrame.pFrameTexture;
    texture->LockRect(0, &LockedRect, NULL, 0);
    if (LockedRect.Pitch != 0)
    {
		memcpy(dest, LockedRect.pBits, width*height*4*sizeof(GLubyte));
    }
    texture->UnlockRect(0);
    sensor->NuiImageStreamReleaseFrame(rgbStream, &imageRGBFrame);
}

void templateMatchingPreprocessing()
{
	FileStorage f;
	cv::Mat templIn, templ;

	// READ - RGB template
	if (!f.isOpened())
	{
		f.open("rgb_32f.xml", FileStorage::READ);
		f["rgb_dif"] >> templIn;
		f.release();
	}

	cv::Rect templRect1(345, 300, 30, 30);
	cv::Mat(templIn, templRect1).copyTo(templ);

	// Set global variable rgbTemplate
	rgbTemplate = templ;

	// READ - depth template 
	if (!f.isOpened())
	{
		f.open("depth_32f.xml", FileStorage::READ);
		f["original_depth"] >> templIn;
		f.release();
	}
	
	cv::Rect templRect2(180, 210, 60, 60);
	cv::Mat(templIn, templRect2).copyTo(templ);

	// Set global variable depthTemplate
	depthTemplate = templ;
}

// Perform the gaussian blur difference on the rgb image (initially, the rgb image is CV_8UC3 - when grabbed from the webcam)
cv::Mat getRGB_GaussianBlurDifference_32F(cv::Mat rgbImage)
{
	cv::Mat grayImage(height, width, CV_32FC1);
	cvtColor(rgbImage, grayImage, CV_RGB2GRAY);
	
	cv::Mat a, b, c, d, result;
	grayImage.convertTo(a, CV_32FC1);

	cv::GaussianBlur(a, b, cv::Size(21,21), 0, 0, 4);
	cv::GaussianBlur(b, c, cv::Size(21,21), 0, 0, 4);
	cv::GaussianBlur(c, d, cv::Size(21,21), 0, 0, 4);
	result = d - c;

	// Result is returned as CV_32FC1
	return result;
}

// Convert a 32F image to 8UC1 - easier to view with imshow
cv::Mat convertToDisplay(cv::Mat inputImage)
{
	double min,max;
	cv::Mat inputImageConv;
	
	minMaxLoc(inputImage, &min, &max);
	inputImage.convertTo(inputImageConv, CV_8UC1, 255.0/max);

	return inputImageConv;
}

// Depth template matching - performed directly on the depth image
bool depthTemplateMatching_32F(cv::Mat depthImage, Point * depthMatchPoint)
{
	double min, max;

	// Check that input image and template have the same type and depth
	assert(depthTemplate.type() == depthImage.type() || depthTemplate.depth() == depthImage.depth());

	int result_cols = depthImage.cols - depthTemplate.cols + 1;
	int result_rows = depthImage.cols - depthTemplate.rows + 1;
	cv::Mat result;
	result.create(result_rows, result_cols, CV_32FC1);

	// Match template only accepts images of type 8U or 32F
	matchTemplate(depthImage, depthTemplate, result, CV_TM_CCOEFF_NORMED);

	// Since we are using CV_TM_COEFF_NORMED -> max value is the one we are looking for
	Point minLoc, maxLoc, matchLoc;
	minMaxLoc(result, &min, &max, &minLoc, &maxLoc, Mat()); 
	matchLoc = maxLoc;

	// Draw a rectangle where the matching is
	rectangle( depthImage, matchLoc, Point( matchLoc.x + depthTemplate.cols , matchLoc.y + depthTemplate.rows ), Scalar::all(1), 2, 8, 0 );

	imshow("depth_matching", convertToDisplay(depthImage));

	(*depthMatchPoint).x = matchLoc.x + depthTemplate.cols/2;
	(*depthMatchPoint).y = matchLoc.y + depthTemplate.rows/2;

	double t = 0.8;
	if (max < t)	return false;
	
	return true;
}

// Perform template matching on the rgb difference - 32F
bool rgbTemplateMatching_32F(cv::Mat rgbDifImage, Point * rgbMatchPoint)
{
	double min, max;

	// Check that the images for template matching have the same type and depth
	assert(rgbTemplate.type() == rgbDifImage.type() || rgbTemplate.depth() == rgbDifImage.depth());

	int result_cols = rgbDifImage.cols - rgbTemplate.cols + 1;
	int result_rows = rgbDifImage.cols - rgbTemplate.rows + 1;
	cv::Mat result;
	result.create(result_rows, result_cols, CV_32FC1);

	// match template only accepts images of type 8U or 32F
	matchTemplate(rgbDifImage, rgbTemplate, result, CV_TM_CCOEFF_NORMED);

	// since we are using CV_TM_COEFF_NORMED -> max value is the one we are looking for
	Point minLoc, maxLoc, matchLoc;
	minMaxLoc(result, &min, &max, &minLoc, &maxLoc, Mat()); 
	matchLoc = maxLoc;

	cv::Mat match_conv = convertToDisplay(rgbDifImage);
	rectangle( match_conv, matchLoc, Point( matchLoc.x + rgbTemplate.cols , matchLoc.y + rgbTemplate.rows ), Scalar::all(255), 2, 8, 0 );
	imshow("rgb_matching", match_conv);

	(*rgbMatchPoint).x = matchLoc.x + rgbTemplate.cols/2;
	(*rgbMatchPoint).y = matchLoc.y + rgbTemplate.rows/2;

	double t = 0.90;
	if (max < t)	return false;

	return true;
}

// Reproject a 3D point to a 2D point
void reproject(const double a[12], double u, double v, double z, double * r) {
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
cv::Mat getDepthColorReconstruction(cv::Mat depthImage, cv::Mat rgbImage, USHORT * data)
{
	cv::Mat coloredDepth(depthImage.size(), CV_8UC3);
	assert(coloredDepth.channels() == 3);
	assert(rgbImage.channels() == 3);

	for (int y=0; y<coloredDepth.rows; ++y)
	{
		for (int x=0; x<coloredDepth.cols; ++x)
		{
			long x_col, y_col;
			USHORT pixelDepth = data[y*width+x];
			USHORT depthInMM = NuiDepthPixelToDepth(pixelDepth);
			
			if (pixelDepth != 0)
			{
				// FOR THIS FUNCTION TO WORK - USE PACKED VERSION OF THE DEPTH
				//NuiImageGetColorPixelCoordinatesFromDepthPixelAtResolution(NUI_IMAGE_RESOLUTION_640x480, NUI_IMAGE_RESOLUTION_640x480, &imageRGBFrame.ViewArea, x, y, pixelDepth, &x_col, &y_col);
				double r[2];
				reproject(aCalib, x, y, (float)(depthInMM/1000.0), r);
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
					coloredDepth.ptr<uchar>(y)[x * 3 + c] = 255; 
				}
			}
		}
	}

	return coloredDepth;
}

// Call calibrator
void callCalibrator()
{
	if (kinectCalibrator.numEntries() == minCalibrationPoints)
	{
		calibResult = kinectCalibrator.calibrate();

		for (unsigned i = 0; i<12; ++i)
		{
			aCalib[i] = calibResult.at<double>(i, 0);
		}
		kinectCalibrator.save();
	}
}

// Provide a debug way for the projections
cv::Mat debugProjections(cv::Mat rgbImage)
{
	typedef std::vector<cv::Point2f> Point2DVector;
	typedef std::vector<cv::Point3f> Point3DVector; 

	Point2DVector projections = kinectCalibrator.projections();
	Point3DVector points3D = kinectCalibrator.points3D();

	for (unsigned i = 0; i < kinectCalibrator.numEntries(); ++i) 
	{
		// Draw projection with greens
		line(rgbImage, Point(projections[i].x - 20, projections[i].y), Point(projections[i].x + 20, projections[i].y), Scalar(0, 255, 0), 1, 8, 0);
		line(rgbImage, Point(projections[i].x, projections[i].y - 20), Point(projections[i].x, projections[i].y + 20), Scalar(0, 255, 0), 1, 8, 0);
	}

	if (kinectCalibrator.numEntries() >= minCalibrationPoints)
	{
		for (unsigned i = 0; i < kinectCalibrator.numEntries(); ++i) 
		{
			// Draw reprojection with red
			double r[2];
			reproject(aCalib, points3D[i].x, points3D[i].y, points3D[i].z, r);
			int x_col = r[0];
			int y_col = r[1];
			line(rgbImage, Point(x_col - 20, y_col), Point(x_col + 20, y_col), Scalar(0, 0, 255), 1, 8, 0);
			line(rgbImage, Point(x_col, y_col - 20), Point(x_col, y_col + 20), Scalar(0, 0, 255), 1, 8, 0);
		}
	}

	return rgbImage;
}

void test_loop()
{
	USHORT data[width*height];// array containing the depth information of each pixel
	getKinectPackedDepthData(data);

	// Get depth data
	cv::Mat original_depth = getDepthImageFromPackedData(data);

	// Get RGB data
	cv::Mat image = getRGBCameraFrame();
	cv::Mat rgbImage;
	cv::flip(image, rgbImage, 1);
	/*cv::Mat rClone = debugProjections(rgbImage);
	imshow("original_image", rClone);
	*/
	if (calibrated == true)
	{
		cv::Mat depthColor = getDepthColorReconstruction(original_depth, rgbImage, data);
		imshow("Depth Color Reconstruction", depthColor);
	}

	if (frameCount % 13 == 0)
	{
		// Template matching
		Point rgbMatchingPoint; 
		Point depthMatchingPoint;
		cv::Mat rgbDif = getRGB_GaussianBlurDifference_32F(rgbImage);
		bool rgbRes = rgbTemplateMatching_32F(rgbDif, &rgbMatchingPoint);
		bool depthRes = depthTemplateMatching_32F(original_depth, &depthMatchingPoint);

		// Test matching in both images
		if (rgbRes == true && depthRes == true)
		{
			// Add these points to the calibrator
			kinectCalibrator.add3DPoint(depthMatchingPoint.x, depthMatchingPoint.y, (NuiDepthPixelToDepth(data[depthMatchingPoint.y * width + depthMatchingPoint.x]))/1000.0);
			kinectCalibrator.addProjCam(rgbMatchingPoint.x, rgbMatchingPoint.y);

			printf("Matching \n");

			callCalibrator();
		}

		frameCount = 1;
	}
	else
	{
		frameCount++;
	}
}

// Main 
int main()
{
	// init Kinect
	if (initKinect() != S_OK)	return -1;

	// init webcam
    if(!(capture = cvCaptureFromCAM(0)))	return -1;

	calibResult = kinectCalibrator.load();

	calibResult = kinectCalibrator.calibrate();
	calibrated = true;
	for (int i = 0; i<12; ++i)
	{
		aCalib[i] = calibResult.at<double>(i, 0);
	}

	// Preprocessing
	templateMatchingPreprocessing();

	// Prepare opencv windows
	cvNamedWindow( "original_image", CV_WINDOW_AUTOSIZE );
	cvNamedWindow( "depth_matching", CV_WINDOW_AUTOSIZE );
	cvNamedWindow( "rgb_matching", CV_WINDOW_AUTOSIZE );
	
	// Main loop
	while ( 1 ) 
	{
		test_loop();

		if ( (cvWaitKey(10) & 255) == 27 ) break;
	}

	// Release the capture device and destroy windows
	cvReleaseCapture( &capture );
	cvDestroyWindow( "original_image" );
	cvDestroyWindow( "depth_matching" );
	cvDestroyWindow( "rgb_matching" );
	
	return 0;
}