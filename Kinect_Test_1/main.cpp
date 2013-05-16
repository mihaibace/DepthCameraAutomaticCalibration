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

#include <vector>
#include "calibrator.h";

using namespace cv;
using namespace std;
using namespace aptarism::vision;

// ----------------------------------------------------------
// Global Variables
// ----------------------------------------------------------
namespace
{
const int width = 640;
const int height = 480;

int toggleNormal = 0;
int saveImage = 0;
double rotate_y = 0; 
double rotate_x = 0;
double trans_x = 0;
double trans_y = 0;
double trans_z = 0;

// Vector holding the calibration points
vector<cv::Point> calibPoints;

// Kinect variables
HANDLE depthStream;				// The indetifier of the Kinect's Depth Camera
HANDLE rgbStream;				// The identifier of the Kinect's RGB Camera
INuiSensor * sensor;            // The kinect sensor

GLubyte rgbData[width*height*4];

// Webcam variables
CvCapture* capture;

// Calibrator 
Calibrator kinectCalibrator;
cv::Mat calibResult;
double aCalib[12];

// Define minimum number of calibration points
int minCalibrationPoints = 6;

// ----------------------------------------------------------
// Function Prototypes
// ----------------------------------------------------------
void draw(void);
void display();
void specialKeys();
void keyboard();

// ----------------------------------------------------------
// OpenGL scene control
// ----------------------------------------------------------
void specialKeys( int key, int x, int y ) 
{ 
	switch (key)
	{
		case GLUT_KEY_RIGHT: rotate_y += 5;
			break;

		case GLUT_KEY_LEFT: rotate_y -= 5;
			break; 

		case GLUT_KEY_UP: rotate_x += 5;
			break;

		case GLUT_KEY_DOWN: rotate_x -= 5;
			break;

		default: 
			break;
	}
 
	//  Request display update
	glutPostRedisplay(); 
}

void keyboard(unsigned char key, int x, int y)
{
	GLdouble modelViewMatrix[16];
	glGetDoublev(GL_MODELVIEW_MATRIX, modelViewMatrix); 

	// x-axis: m0, m1, m2
	// y-axis: m4, m5, m6
	// z-axis: m8, m9, m10

	int scalingFactor = 1;

	switch(key)
	{
	case 'a': // Y Axis
			trans_x += modelViewMatrix[0] * scalingFactor;
			trans_y += modelViewMatrix[1] * scalingFactor;
			trans_z += modelViewMatrix[2] * scalingFactor;
		break;

	case 'd': 
			trans_x -= modelViewMatrix[0] * scalingFactor;
			trans_y -= modelViewMatrix[1] * scalingFactor;
			trans_z -= modelViewMatrix[2] * scalingFactor;
		break;

	case 'w': // X Axis
			trans_x -= modelViewMatrix[4] * scalingFactor;
			trans_y -= modelViewMatrix[5] * scalingFactor;
			trans_z -= modelViewMatrix[6] * scalingFactor;
		break;

	case 's': 
			trans_x += modelViewMatrix[4] * scalingFactor;
			trans_y += modelViewMatrix[5] * scalingFactor;
			trans_z += modelViewMatrix[6] * scalingFactor;
		break;

	case 'z': // Z Axis
			trans_x -= modelViewMatrix[8] * scalingFactor;
			trans_y -= modelViewMatrix[9] * scalingFactor;
			trans_z -= modelViewMatrix[10] * scalingFactor;
		break;

	case 'x': 
			trans_x += modelViewMatrix[8] * scalingFactor;
			trans_y += modelViewMatrix[9] * scalingFactor;
			trans_z += modelViewMatrix[10] * scalingFactor;
		break;

	case 't': 
			if (toggleNormal == 1) 
				toggleNormal = 0; 
			else 
				toggleNormal = 1;
		break;

	case 'p': saveImage = 1;
		break;

	default:
		break;
	}

	glutPostRedisplay();
}

// ----------------------------------------------------------
// OpenGL init window
// ----------------------------------------------------------
bool init(int argc, char* argv[]) 
{
	// Vector for calibration points init
	calibPoints.reserve(1000);

	// OpenGL init
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(width,height);
    glutCreateWindow("Kinect 3D Point Cloud");
	glutDisplayFunc(draw);
    glutIdleFunc(draw);
	glutSpecialFunc(specialKeys);
	glutKeyboardFunc(keyboard);
	

	return true;
}

// ----------------------------------------------------------
// Initialize Kinect
// ----------------------------------------------------------
bool initKinect() 
{
    // Get a working kinect sensor
    int numSensors;
    if (NuiGetSensorCount(&numSensors) < 0 || numSensors < 1) return false;
    if (NuiCreateSensorByIndex(0, &sensor) < 0) return false;

    // Initialize sensor
    sensor->NuiInitialize(NUI_INITIALIZE_FLAG_USES_DEPTH | NUI_INITIALIZE_FLAG_USES_COLOR);
	
	// Set the camera as a Depth Camera
	sensor-> NuiImageStreamOpen(
		NUI_IMAGE_TYPE_DEPTH, // Depth camera
		NUI_IMAGE_RESOLUTION_640x480, // Image resolution
		0, // Image stream flags, e.g. near mode
		2, // Number of frames to buffer
		NULL, // Event handle
		&depthStream);

	// Set the camera as a RGB Camera
	sensor-> NuiImageStreamOpen(
		NUI_IMAGE_TYPE_COLOR, // Depth camera
		NUI_IMAGE_RESOLUTION_640x480, // Image resolution
		0, // Image stream flags, e.g. near mode
		2, // Number of frames to buffer
		NULL, // Event handle
		&rgbStream);


    return sensor;
}

// ----------------------------------------------------------
// Get data from the Kinect
// ----------------------------------------------------------
void getKinectDepthData(USHORT * dest) 
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


NUI_IMAGE_FRAME imageRGBFrame;
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


// ----------------------------------------------------------
// Draw Kinect data
// ----------------------------------------------------------

cv::Mat getRGB_GaussianBlurDifference(cv::Mat rgbImage) 
{
	cv::Mat grayImage(height, width, CV_8UC1);
	cvtColor(rgbImage, grayImage, CV_RGB2GRAY);
	//cv::imshow("RGB: Grayscale Image", grayImage);

	cv::Mat gray_float1, gray_float2;
	grayImage.convertTo(gray_float1, CV_32FC1);

	cv::Mat b, c, d, result, result2;
	cv::GaussianBlur(gray_float1, b, cv::Size(21,21), 0, 0, 4);
	cv::GaussianBlur(b, c, cv::Size(21,21), 0, 0, 4);
	cv::GaussianBlur(c, d, cv::Size(21,21), 0, 0, 4);
	result = d - c;

	double min_val, max_val;
	cv::minMaxLoc(result, &min_val, &max_val);
	result.convertTo(result2, CV_8UC1, 255.0/max_val);
	//result.convertTo(result2, CV_8UC1, 10);

	// Result 2 is returned as CV_8UC1
	return result2;
}

// Perform the gaussian blur difference on the rgb image
// Initially, the rgb image is CV_8UC3 - when grabbed from the webcam
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

// Try to remove all the saturated pixels (white, value = 255)
cv::Mat filterDepthSaturatedPixels(cv::Mat matInput)
{
	double min_val, max_val;
	cv::Mat result = matInput.clone();

	cv::minMaxLoc(matInput, &min_val, &max_val);

	for (int y=0; y<matInput.rows; ++y)
	{
		for (int x=0; x<matInput.cols; ++x)
		{
			int value = (int) matInput.at<uchar>(y, x);
			if (value == max_val)
			{
				result.at<uchar>(y, x) = 0;
			}
			else
			{
				result.at<uchar>(y,x) = matInput.at<uchar>(y, x);
			}
		}
	}

	return result;
}

// Compute the difference between succesive Gaussian filters on the depth image
cv::Mat getDepth_GaussianBlurDifference(USHORT * data)
{
	// OpenCV blurring
	cv::Mat a(height, width, CV_16SC1, data);
	cv::Mat a_float, a_float2;
	a.convertTo(a_float, CV_32FC1);
	cv::Mat b, c, d, result;
	cv::GaussianBlur(a_float, b, cv::Size(21,21), 0, 0, 4);
	cv::GaussianBlur(b, c, cv::Size(21,21), 0, 0, 4);
	cv::GaussianBlur(c, d, cv::Size(21,21), 0, 0, 4);
	result = d - c;

	cv::Mat scaled_result;
	double min_val, max_val;
	cv::minMaxLoc(result, &min_val, &max_val);
	result.convertTo(scaled_result, CV_8UC1, 10);
	//result.convertTo(scaled_result, CV_8UC1, 255.0/max_val);
	
	return scaled_result;
}	

cv::Mat getDepth_GaussianBlurDifference_32F(USHORT * data)
{
	// init a
	USHORT * data2 = (USHORT *) malloc(height * width * sizeof(USHORT));
	for (int y = 0; y < height; ++y)
	{
		for (int x = 0; x < width; ++x)
		{
			// Populate the matrix with actual depth in mm
			data2[y * width + x] = NuiDepthPixelToDepth(data[y * width + x]);
		}
	}

	cv::Mat b, c, d, result, a_float;

	cv::Mat a(height, width, CV_16SC1, data2);
	a.convertTo(a_float, CV_32FC1);
	
	cv::GaussianBlur(a_float, b, cv::Size(21,21), 0, 0, 4);
	cv::GaussianBlur(b, c, cv::Size(21,21), 0, 0, 4);
	cv::GaussianBlur(c, d, cv::Size(21,21), 0, 0, 4);

	result = d - c;
	// result is CV_32FC1
	return result;
}

void getGradientMagnitudeAndOrientation(cv::Mat matInput, cv::Mat * gradient_mag, cv::Mat * gradient_orientation)
{
	// Compute the gradient of the image
	Mat gradX, gradY, abs_gradX, abs_gradY, gradient, approx_gradient, abs_gradient;

	// Gradient on X direction
	Sobel(matInput, gradX, CV_16S, 1, 0, 3, 1, 0, 4);
	convertScaleAbs(gradX, abs_gradX);

	// Gradient on Y direction
	Sobel(matInput, gradY, CV_16S, 0, 1, 3, 1, 0, 4);
	convertScaleAbs(gradY, abs_gradY);

	// Total gradient
	Mat magnitude, abs_magnitude, scaled_orientation, orientation, abs_orientation, abs_gradX_f, abs_gradY_f;
	abs_gradX.convertTo(abs_gradX_f, CV_32FC1);
	abs_gradY.convertTo(abs_gradY_f, CV_32FC1);
	cartToPolar(abs_gradX_f, abs_gradY_f, magnitude, orientation, true);

	// Display gradient
	convertScaleAbs(magnitude, abs_magnitude);

	//convertScaleAbs(orientation, abs_orientation);
	orientation.convertTo(scaled_orientation, CV_8UC1, 255.0/360);

	*gradient_mag = abs_magnitude.clone();
	*gradient_orientation = scaled_orientation.clone();
}

/*
void heatmap()
{
	// Try to display a color map of the gradient orientation
	//IplImage *img = cvLoadImage(argv[1],CV_LOAD_IMAGE_GRAYSCALE); //the input image, grayscale
	Mat heatmap; //the heatmap image, color
	resize(imread("colors.jpg",CV_LOAD_IMAGE_COLOR), heatmap, Size(1,256));
	Mat dst(scaled_orientation.size(), CV_8UC3); //the result image, color

	Mat scaled_orientation3(scaled_orientation.size(), CV_8UC3);
	int from_to[] = {0,0, 0,1, 0,2};
	mixChannels(&scaled_orientation, 1, &scaled_orientation3, 1, from_to, 3);
	LUT(scaled_orientation3, heatmap.col(0), dst);

	//cv::namedWindow("Blur_difference_gradient_orientation_heatmap", CV_WINDOW_AUTOSIZE );
	cv::Mat heat_map(dst);
	//cv::imshow("Blur_difference_gradient_orientation_heatmap", heat_map);
	//cvSaveImage("false_colors.bmp",dst);
}
*/

cv::Mat findLocalMaximCorrelation(cv::Mat depthImage, cv::Mat rgbImage)
{
	cv::Mat result = depthImage.clone();

	double depth_min_val, depth_max_val, rgb_min_val, rgb_max_val;

	cv::minMaxLoc(depthImage, &depth_min_val, &depth_max_val);
	cv::minMaxLoc(rgbImage, &rgb_min_val, &rgb_max_val);

	for (int y=0; y<depthImage.rows; ++y)
	{
		for (int x=0; x<depthImage.cols; ++x)
		{
			int depthValue = (int) depthImage.at<uchar>(y, x);
			float rgbValue = rgbImage.at<float>(y, x);
			if (depthValue == ((int) depth_max_val) && rgbValue == ((float) rgb_max_val))
			{
				result.at<uchar>(y, x) = depthImage.at<uchar>(y, x);
			}
			else
			{
				result.at<uchar>(y,x) = 0;
			}
		}
	}

	return result;
}

// Recontruct the depth image from the color image
cv::Mat getDepthColorReconstruction(cv::Mat depthImage, cv::Mat rgbImage, USHORT * data)
{
	cv::Mat coloredDepth(depthImage.size(), CV_8UC4);
	assert(coloredDepth.channels() == 4);

	for (int y=0; y<coloredDepth.rows; ++y)
	{
		for (int x=0; x<coloredDepth.cols; ++x)
		{
			long x_col, y_col;
			USHORT pixelDepth = data[y*width+x];
			
			if (pixelDepth != 0)
			{
				// FOR THIS FUNCTION TO WORK - USE PACKED VERSION OF THE DEPTH
				NuiImageGetColorPixelCoordinatesFromDepthPixelAtResolution(NUI_IMAGE_RESOLUTION_640x480, NUI_IMAGE_RESOLUTION_640x480, &imageRGBFrame.ViewArea, x, y, pixelDepth, &x_col, &y_col);
				//NuiImageGetColorPixelCoordinatesFromDepthPixel(NUI_IMAGE_RESOLUTION_640x480, NULL, x, y, pixelDepth, &x_col, &y_col); - NOT WORKING WELL, NEED TO USE THE ONE ABOVE

				if ((x_col >= 0 && x_col < width) && (y_col >= 0 && y_col < height))
				{
					// take the color from x_col and y_col and project it in the depth image
				
					// iterate through all the channels
					for (int c = 0; c < 4; ++c) 
					{
						coloredDepth.ptr<uchar>(y)[x * 4 + c] = rgbImage.ptr<uchar>(y_col)[x_col * 4 + c]; 
						//coloredDepth.data[coloredDepth.channels()*(coloredDepth.cols*y + x) + c] = rgbImage.data[rgbImage.channels()*(rgbImage.cols*y_col + x_col) + c];
					}
				}
				else
				{
					for (int c = 0; c < 4; ++c) 
					{
						coloredDepth.ptr<uchar>(y)[x * 4 + c] = 0; 
						//coloredDepth.data[coloredDepth.channels()*(coloredDepth.cols*y + x) + c] = 0;
					}
				}
			}
			else
			{
				// We have no depth information about this points (depth from kinect = 0)
				for (int c = 0; c < 4; ++c) 
				{
					coloredDepth.ptr<uchar>(y)[x * 4 + c] = 255; 
					//coloredDepth.data[coloredDepth.channels()*(coloredDepth.cols*y + x) + c] = 0;
				}
			}
		}
	}

	return coloredDepth;
}

Point depthTemplateMatching(cv::Mat depthDifImage)
{
	// create a copy of the depth image
	cv::Mat depth_copy;
	depthDifImage.copyTo(depth_copy);

	cv::Mat templIN = imread("depth_template_3.jpg", 0); // 0 means grayscale image; 1 would take as 3 channel colour image 
	cv::Mat templ;
	templIN.convertTo(templ, CV_8UC1);

	// convert depth image to be of the same type
	depthDifImage.convertTo(depth_copy, CV_8UC1);

	int result_cols = depth_copy.cols - templ.cols + 1;
	int result_rows = depth_copy.cols - templ.rows + 1;
	cv::Mat result;
	result.create(result_rows, result_cols, CV_8UC1);

	// type - 0 = 8U, 1 = 8S, 2 = 16U, 3 = 16S, 4 = 32S, 5 = 32F, 6 = 64F
	assert(templ.type() == depth_copy.type() || templ.depth() == depth_copy.depth());

	// match template only accepts images of type 8U or 32F
	matchTemplate(depth_copy, templ, result, CV_TM_CCOEFF_NORMED);


	//normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat());

	double minVal, maxVal;
	Point minLoc, maxLoc, matchLoc;

	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat()); 

	// since we are using CV_TM_COEFF_NORMED -> max value is the one we are looking for
	matchLoc = maxLoc;

	rectangle( depth_copy, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar::all(255), 2, 8, 0 );
	line(result, Point(matchLoc.x - 20, matchLoc.y), Point(matchLoc.x + 20, matchLoc.y), Scalar::all(1), 1, 8, 0);
	line(result, Point(matchLoc.x, matchLoc.y - 20), Point(matchLoc.x, matchLoc.y + 20), Scalar::all(1), 1, 8, 0);

	cv::Mat result_conv;
	double min, max;
	minMaxLoc(result, &min, &max);
	result.convertTo(result_conv, CV_8UC1, 255.0/max);

	imshow("Depth image matching", depth_copy);
	//imshow("Depth matching result", result);

	//if (saveImage == 1)
	//{
	//	imwrite("template_match.jpg", result_conv);
	//	imwrite("depth_dif_match.jpg", depth_copy);
	//	saveImage = 0;
	//}

	Point depthMatchPoint;
	depthMatchPoint.x = matchLoc.x + templ.cols/2;
	depthMatchPoint.y = matchLoc.y + templ.rows/2;
	return depthMatchPoint;
}

// Template matching directly on the depth image
Point depthTemplateMatching_32F(cv::Mat depthImage, int scale)
{
	double min, max;

	FileStorage f;
	cv::Mat templIn, templ;
	if (!f.isOpened())
	{
		f.open("depth_32f.xml", FileStorage::READ);
		f["original_depth"] >> templIn;
		f.release();

		// NEVER DRAW THE RECTANGLE ON THE IMAGE!!!!
		//rectangle( templIn, Point(180,210), Point(240, 270), Scalar::all(1), 2, 8, 0 );
		minMaxLoc(templIn, &min, &max);
		cv::Mat templIn_conv;
		templIn.convertTo(templIn_conv, CV_8UC1, 255.0/max);
		//imshow("Template image", templIn_conv);
	}

	
	int resize = 10;
	cv::Rect templRect(180 + scale * 10, 210 + scale * 10, 60 - scale * 10, 60 - scale * 10);
	cv::Mat(templIn, templRect).copyTo(templ);

	minMaxLoc(templ, &min, &max);
	cv::Mat templ_conv;
	templ.convertTo(templ_conv, CV_8UC1, 255.0/(max-min), -min*255.0/(max-min));
	imshow("Template rectangle cropped", templ_conv);

	f.open("depth_template_cropped.xml", FileStorage::WRITE);
	f << "depth_template" << templ;
	f.release();

	// create a copy of the depth image
	cv::Mat depth_copy;
	depthImage.copyTo(depth_copy);

	// type - 0 = 8U, 1 = 8S, 2 = 16U, 3 = 16S, 4 = 32S, 5 = 32F, 6 = 64F
	assert(templ.type() == depth_copy.type() || templ.depth() == depth_copy.depth());

	int result_cols = depth_copy.cols - templ.cols + 1;
	int result_rows = depth_copy.cols - templ.rows + 1;
	cv::Mat result;
	result.create(result_rows, result_cols, CV_32FC1);

	// match template only accepts images of type 8U or 32F
	matchTemplate(depth_copy, templ, result, CV_TM_CCOEFF_NORMED);

	//normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat());

	double minVal, maxVal;
	Point minLoc, maxLoc, matchLoc;

	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat()); 
	//printf("Matching value = %f \n", maxVal);
	// since we are using CV_TM_COEFF_NORMED -> max value is the one we are looking for
	matchLoc = maxLoc;

	rectangle( depth_copy, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar::all(1), 2, 8, 0 );
	line(result, Point(matchLoc.x - 20, matchLoc.y), Point(matchLoc.x + 20, matchLoc.y), Scalar::all(1), 2, 8, 0);
	line(result, Point(matchLoc.x, matchLoc.y - 20), Point(matchLoc.x, matchLoc.y + 20), Scalar::all(1), 2, 8, 0);

	cv::Mat result_conv, depth_conv;
	
	minMaxLoc(result, &min, &max);
	result.convertTo(result_conv, CV_8UC1, 255.0/max);

	minMaxLoc(depth_copy, &min, &max);
	depth_copy.convertTo(depth_conv, CV_8UC1, 255.0/max);

	imshow("Depth image matching", depth_conv);
	//imshow("Depth matching result", result_conv);

	Point depthMatchPoint;
	depthMatchPoint.x = matchLoc.x + templ.cols/2;
	depthMatchPoint.y = matchLoc.y + templ.rows/2;

	double t = 0.78 + scale * 0.1;
	if (maxVal >= t)
	{
		return depthMatchPoint;
	}
	else
	{
		return Point(-100,-100);
	}
}

Point rgbTemplateMatching(cv::Mat rgbDifImage)
{
	// create a copy of the rgb image
	cv::Mat rgb_copy;
	rgbDifImage.copyTo(rgb_copy);

	cv::Mat templIN = imread("rgb_template.jpg", 0); // 0 means grayscale image; 1 would take as 3 channel colour image 
	cv::Mat templ;
	templIN.convertTo(templ, CV_8UC1);

	// convert depth image to be of the same type
	rgbDifImage.convertTo(rgb_copy, CV_8UC1);

	int result_cols = rgb_copy.cols - templ.cols + 1;
	int result_rows = rgb_copy.cols - templ.rows + 1;
	cv::Mat result;
	result.create(result_rows, result_cols, CV_8UC1);

	// 0 = 8U, 1 = 8S, 2 = 16U, 3 = 16S, 4 = 32S, 5 = 32F, 6 = 64F
	assert(templIN.type() == rgb_copy.type() || templIN.depth() == rgb_copy.depth());
	
	// match template only accepts images of type 8U or 32F
	matchTemplate(rgb_copy, templ, result, CV_TM_CCOEFF_NORMED);

	//normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat());

	double minVal, maxVal;
	Point minLoc, maxLoc, matchLoc;

	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat()); 

	// since we are using CV_TM_COEFF_NORMED -> max value is the one we are looking for
	matchLoc = maxLoc;

	rectangle( rgb_copy, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar::all(255), 2, 8, 0 );
	line(result, Point(matchLoc.x - 20, matchLoc.y), Point(matchLoc.x + 20, matchLoc.y), Scalar::all(1), 1, 8, 0);
	line(result, Point(matchLoc.x, matchLoc.y - 20), Point(matchLoc.x, matchLoc.y + 20), Scalar::all(1), 1, 8, 0);

	cv::Mat result_conv;
	double min, max;
	minMaxLoc(result, &min, &max);
	result.convertTo(result_conv, CV_8UC1, 255.0/max);

	//imshow("RGB image matching", rgb_copy);
	//imshow("Result", result);

	/*if (saveImage == 1)
	{
		imwrite("rgb_template_match.jpg", result_conv);
		imwrite("rgb_dif_match.jpg", rgb_copy);
		saveImage = 0;
	}*/

	// Return the position of the matched point
	Point rgbMatchPoint;
	rgbMatchPoint.x = matchLoc.x + templ.cols/2;
	rgbMatchPoint.y = matchLoc.y + templ.rows/2;
	return rgbMatchPoint;
}

// Perform template matching on the rgb difference - 32F
Point rgbTemplateMatching_32F(cv::Mat rgbDifImage)
{
	double min, max;
	
	FileStorage f;
	cv::Mat templIn, templ;
	if (!f.isOpened()){
		f.open("rgb_32f.xml", FileStorage::READ);
		f["rgb_dif"] >> templIn;
		f.release();

		// NEVER DRAW RECTANGLE ON THE IMAGE
		//rectangle( templIn, Point(345,300), Point(375, 330), Scalar::all(1), 2, 8, 0 );
		double min, max;
		minMaxLoc(templIn, &min, &max);
		cv::Mat templIn_conv;
		templIn.convertTo(templIn_conv, CV_8UC1, 255.0/max);
		//imshow("rgb template", templIn_conv);
	}

	cv::Rect templRect(345, 300, 30, 30);
	cv::Mat(templIn, templRect).copyTo(templ);

	minMaxLoc(templ, &min, &max);
	cv::Mat templ_conv;

	// Display the template
	templ.convertTo(templ_conv, CV_8UC1, 255.0/(max-min), -min*255.0/(max-min));
	imshow("RGB Template rectangle cropped", templ_conv);

	f.open("rgb_template_cropped.xml", FileStorage::WRITE);
	f << "rgb_template" << templ;
	f.release();

	// create a copy of the depth image
	cv::Mat rgb_copy;
	rgbDifImage.copyTo(rgb_copy);

	// type - 0 = 8U, 1 = 8S, 2 = 16U, 3 = 16S, 4 = 32S, 5 = 32F, 6 = 64F
	assert(templ.type() == rgb_copy.type() || templ.depth() == rgb_copy.depth());

	int result_cols = rgb_copy.cols - templ.cols + 1;
	int result_rows = rgb_copy.cols - templ.rows + 1;
	cv::Mat result;
	result.create(result_rows, result_cols, CV_32FC1);

	// match template only accepts images of type 8U or 32F
	matchTemplate(rgb_copy, templ, result, CV_TM_CCOEFF_NORMED);

	//normalize( result, result, 0, 1, NORM_MINMAX, -1, Mat());

	double minVal, maxVal;
	Point minLoc, maxLoc, matchLoc;

	minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat()); 
	//printf("Matching value = %f \n", maxVal);
	// since we are using CV_TM_COEFF_NORMED -> max value is the one we are looking for
	matchLoc = maxLoc;

	rectangle( rgb_copy, matchLoc, Point( matchLoc.x + templ.cols , matchLoc.y + templ.rows ), Scalar::all(1), 2, 8, 0 );
	//line(result, Point(matchLoc.x - 20, matchLoc.y), Point(matchLoc.x + 20, matchLoc.y), Scalar::all(1), 2, 8, 0);
	//line(result, Point(matchLoc.x, matchLoc.y - 20), Point(matchLoc.x, matchLoc.y + 20), Scalar::all(1), 2, 8, 0);

	cv::Mat result_conv, rgb_conv;
	
	minMaxLoc(result, &min, &max);
	result.convertTo(result_conv, CV_8UC1, 255.0/max);

	minMaxLoc(rgb_copy, &min, &max);
	rgb_copy.convertTo(rgb_conv, CV_8UC1, 255.0/max);

	imshow("RGB image matching", rgb_conv);
	//imshow("RGB matching result", result_conv);

	Point rgbMatchPoint;
	rgbMatchPoint.x = matchLoc.x + templ.cols/2;
	rgbMatchPoint.y = matchLoc.y + templ.rows/2;

	double t = 0.90;
	if (maxVal >= t)
	{
		return rgbMatchPoint;
	}
	else
	{
		return Point(-100,-100);
	}
}

cv::Mat getRGBCameraFrame()
{

    Mat frame, frameCopy, image;
    //cvNamedWindow("result", 1);

    if(capture)
    {
		IplImage* iplImg = cvQueryFrame(capture);
		frame = iplImg;
		//imshow("result", frame);
		
		/*
		if(!frame.empty())
		{
			if(waitKey( 10 ) >= 0)
				cvReleaseCapture(&capture);

			
			//waitKey(0);
		}*/
	}

	return frame;
}

cv::Mat getDepthImageFromPackedData(USHORT * data)
{
		// init a
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

// take number image type number (from cv::Mat.type()), get OpenCV's enum string.
string getImgType(int imgTypeInt)
{
    int numImgTypes = 35; // 7 base types, with five channel options each (none or C1, ..., C4)

    int enum_ints[] =       {CV_8U,  CV_8UC1,  CV_8UC2,  CV_8UC3,  CV_8UC4,
                             CV_8S,  CV_8SC1,  CV_8SC2,  CV_8SC3,  CV_8SC4,
                             CV_16U, CV_16UC1, CV_16UC2, CV_16UC3, CV_16UC4,
                             CV_16S, CV_16SC1, CV_16SC2, CV_16SC3, CV_16SC4,
                             CV_32S, CV_32SC1, CV_32SC2, CV_32SC3, CV_32SC4,
                             CV_32F, CV_32FC1, CV_32FC2, CV_32FC3, CV_32FC4,
                             CV_64F, CV_64FC1, CV_64FC2, CV_64FC3, CV_64FC4};

    string enum_strings[] = {"CV_8U",  "CV_8UC1",  "CV_8UC2",  "CV_8UC3",  "CV_8UC4",
                             "CV_8S",  "CV_8SC1",  "CV_8SC2",  "CV_8SC3",  "CV_8SC4",
                             "CV_16U", "CV_16UC1", "CV_16UC2", "CV_16UC3", "CV_16UC4",
                             "CV_16S", "CV_16SC1", "CV_16SC2", "CV_16SC3", "CV_16SC4",
                             "CV_32S", "CV_32SC1", "CV_32SC2", "CV_32SC3", "CV_32SC4",
                             "CV_32F", "CV_32FC1", "CV_32FC2", "CV_32FC3", "CV_32FC4",
                             "CV_64F", "CV_64FC1", "CV_64FC2", "CV_64FC3", "CV_64FC4"};

    for(int i=0; i<numImgTypes; i++)
    {
        if(imgTypeInt == enum_ints[i]) return enum_strings[i];
    }
    return "unknown image type";
}

void reproject(const double a[12], double u, double v, double z, double r[2]) {
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

void drawKinectPointCloud()
{
	//  Clear screen and Z-buffer
	glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);

	// Reset transformations
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
 
	// Rotate when user changes rotate_x and rotate_y
	glRotatef( rotate_x, 1.0, 0.0, 0.0 );
	glRotatef( rotate_y, 0.0, 1.0, 0.0 );

	// Translate the image
	glTranslatef(-trans_x, -trans_y, -trans_z);
	
	// Scale all the coordinates: for visualisation purposes
	glScalef(1.5, 1.5, 1.5);     

	/* Get Kinect Depth Data */ 

	USHORT data[width*height];  // array containing the depth information of each pixel
	getKinectDepthData(data);


	/* RGB Template Matching */
	cv::Mat image = getRGBCameraFrame();
	cv::Mat rgbImage;
	cv::flip(image, rgbImage, 1);
	imshow("Original image", rgbImage);
	
	cv::Mat rgbDif = getRGB_GaussianBlurDifference_32F(rgbImage);

	Point rgbMatchingPoint = rgbTemplateMatching_32F(rgbDif);
	
	/* Depth Template Matching */
	cv::Mat original_depth = getDepthImageFromPackedData(data);
	//cv::Mat depthDif = getDepth_GaussianBlurDifference_32F(data);
	Point depthMatchLoc = depthTemplateMatching_32F(original_depth, 0);
	if (depthMatchLoc.x != -100 && depthMatchLoc.y != -100)
	{
		depthMatchLoc = depthTemplateMatching_32F(original_depth, 1);
	}

	/* Check if matching is found in both pictures */ 
	if ((rgbMatchingPoint.x != -100 && rgbMatchingPoint.y != -100) && (depthMatchLoc.x != -100 && depthMatchLoc.y != -100))
	{
		printf("Matching \n");
		// Add these points to the calibrator
		kinectCalibrator.add3DPoint(depthMatchLoc.x, depthMatchLoc.y, NuiDepthPixelToDepth(data[depthMatchLoc.y * width + depthMatchLoc.x]));
		kinectCalibrator.addProjCam(rgbMatchingPoint.x, rgbMatchingPoint.y);
	}
	else
	{
		printf("NO Matching \n");
	}
						
	if (kinectCalibrator.numEntries() == 12)
	{
		calibResult = kinectCalibrator.calibrate();
		for (int i = 0; i<12; ++i)
		{
			aCalib[i] = calibResult.at<double>(i, 0);
		}

		kinectCalibrator.save();
	}

	///cv::Mat coloredDepth = getDepthColorReconstruction(depthImage, rgbImage, data);
	//cv::Mat depthDif = getDepth_GaussianBlurDifference(data);
	
	/*
	//cv::Mat gradMag, gradOrient;
	//getGradientMagnitudeAndOrientation(depthDif, &gradMag, &gradOrient);
	//imshow("Depth: Gradient Magnitude", gradMag);
	//imshow("Depth: Gradient Orientation", gradOrient);

	cv::Mat rgbDif = getRGB_GaussianBlurDifference(rgbImage);
	//imshow("RGB: Gaussian Blur Difference", rgbDif);

	Point depthMatchLoc = depthTemplateMatching(depthDif);
	Point rgbMatchLoc = rgbTemplateMatching(rgbDif);
	long rgbX, rgbY;
	USHORT matchDepth = data[depthMatchLoc.y*width+depthMatchLoc.x];
	NuiImageGetColorPixelCoordinatesFromDepthPixelAtResolution(NUI_IMAGE_RESOLUTION_640x480, NUI_IMAGE_RESOLUTION_640x480, &imageRGBFrame.ViewArea, depthMatchLoc.x, depthMatchLoc.y, matchDepth, &rgbX, &rgbY);


	// Add matched point to array
	if (abs(rgbX - rgbMatchLoc.x) < 10 && abs(rgbY - rgbMatchLoc.y) < 10)
	{
		calibPoints.push_back(rgbMatchLoc);
	}

	// iterate thorugh all the points and display them
	for (int i=0; i < calibPoints.size(); i++)
	{
		line(rgbImage, Point(calibPoints[i].x - 20, calibPoints[i].y), Point(calibPoints[i].x + 20, calibPoints[i].y), Scalar(0,0,255), 2, 8, 0); // image is BGRA
		line(rgbImage, Point(calibPoints[i].x, calibPoints[i].y - 20), Point(calibPoints[i].x, calibPoints[i].y + 20), Scalar(0,0,255), 2, 8, 0); 
	}
	imshow("Webcam image - correspondences", rgbImage);

	*/

	//cv::Mat filtered_depthDif = filterDepthSaturatedPixels(depthDif);
	//imshow("Depth: Filtered Gaussian Blur Difference", filtered_depthDif);

	//cv::Mat localMaximaCorrelation = findLocalMaximCorrelation(depthDif, rgbDif);
	//imshow("Correlation result", localMaximaCorrelation);

	// Display the 3D Coloring only when there are at least "minCalibrationPoints" calibration points
	if (kinectCalibrator.numEntries() >= 20)
	{
		// Display the points as a 3D point cloud
		glBegin(GL_POINTS);
			for (int y = 0; y < height; y+=2)
			{
				USHORT *line = data + y * width;
				for (int x = 0; x < width; x+=2)
				{
					// Packed pixel depth
					USHORT pixelDepth = line[x];
					// Unpack pixel to get the depth in MM
					USHORT depthInMM = NuiDepthPixelToDepth(line[x]);

					// Get 3D coordinates from Depth Image Space
					Vector4 pointToDisplay = NuiTransformDepthImageToSkeleton(x, y, depthInMM);

					long x_col, y_col;
					if (pixelDepth != 0)
					{
						//NuiImageGetColorPixelCoordinatesFromDepthPixelAtResolution(NUI_IMAGE_RESOLUTION_640x480, NUI_IMAGE_RESOLUTION_640x480, &imageRGBFrame.ViewArea, x, y, pixelDepth, &x_col, &y_col);
						double r[2];
						
						reproject(aCalib,x,y,depthInMM, r);
						x_col = r[0];
						y_col = r[1];

						if ((x_col >= 0 && x_col < width) && (y_col >= 0 && y_col < height))
						{
							// take the color from x_col and y_col and project it in the depth image
							int blue = (int) rgbImage.ptr<uchar>(y_col)[x_col * 4 + 0]; 
							int green = (int) rgbImage.ptr<uchar>(y_col)[x_col * 4 + 1]; 
							int red = (int) rgbImage.ptr<uchar>(y_col)[x_col * 4 + 2]; 
						
							// Vertex Color
							glColor3f((float)red/255.0, (float)green/255.0, (float)blue/255.0);
							glVertex3f(pointToDisplay.x, pointToDisplay.y, pointToDisplay.z);
						}
					}				
				}
			}
		glEnd();
	}
	

	// Toggle variable that allows to have a button that controls the display of the normal vectors
	if (toggleNormal == 1)
	{
		// Normal estimation: try to find 4 connected neighbours and estimate the normal based on them
		for (int y = 0; y < height; y+=10)
		{
			USHORT *line = data + y * width;
			for (int x = 0; x < width; x+=10)
			{
				// Verify that all the neighbours are within bounds
				if ((x-1 >= 0) && (x+1 <= width) && (y-1 >= 0) && (y+1 <= height))
				{
					// Packed pixel depth
					USHORT pixelDepth = line[x];
					// Unpack pixel to get the depth in MM
					USHORT depthInMM = NuiDepthPixelToDepth(line[x]);
					
					Vector4 pointOfInterest = NuiTransformDepthImageToSkeleton(x, y, depthInMM);
					Vector4 nLeft = NuiTransformDepthImageToSkeleton(x-1, y, NuiDepthPixelToDepth(line[x-1]));
					Vector4 nRight = NuiTransformDepthImageToSkeleton(x+1, y, NuiDepthPixelToDepth(line[x+1]));
					Vector4 nUp = NuiTransformDepthImageToSkeleton(x, y-1, NuiDepthPixelToDepth(line[x-width]));
					Vector4 nDown = NuiTransformDepthImageToSkeleton(x, y+1, NuiDepthPixelToDepth(line[x+width]));

					// We must verify that all the neighbours have depth information
					// This is verified by having a depth different from 0
					if ((nLeft.z != 0) && (nRight.z != 0) && (nUp.z != 0) && (nDown.z != 0) && (pointOfInterest.z != 0)) 
					{
						// We must verify that the neighbours have similar depth, with a specific delta (e.g. delta = 10)
						int delta = 10;

						if ((abs(line[x-1]-line[x]) < delta) && (abs(line[x+1]-line[x]) < delta) && 
							(abs(line[x-width]-line[x]) < delta) && (abs(line[x+width]-line[x]) < delta))
						{
							// Vector nLeft-nRight
							float a1, a2, a3;
							a1 = nLeft.x - nRight.x;
							a2 = nLeft.y - nRight.y;
							a3 = nLeft.z - nRight.z;

							// Vector nUp-nDown
							float b1, b2, b3;
							b1 = nUp.x - nDown.x;
							b2 = nUp.y - nDown.y;
							b3 = nUp.z - nDown.z;

							// Compute the cross product between the 2 vectors to get the normal vector
							float c1, c2, c3;
							c1 = a2 * b3 - a3 * b2;
							c2 = a3 * b1 - a1 * b3;
							c3 = a1 * b2 - a2 * b1;

							// Normalisation of vector c
							float normC = sqrt(c1 * c1 + c2 * c2 + c3 * c3);

							c1 = c1/normC;
							c2 = c2/normC;
							c3 = c3/normC;

							assert(_finite(c1));
							assert(_finite(c2));
							assert(_finite(c3));

							// s is a scaling factor. 
							// The parameters c1, c2 and c3 have very small values that are not visible when visualising. 
							long s = 1;
							glBegin(GL_LINES);
								glColor3f(0.0, 1.0, 0.0);
								glVertex3f(pointOfInterest.x, pointOfInterest.y, pointOfInterest.z);
								glVertex3f(pointOfInterest.x + s*c1, pointOfInterest.y + s*c2, pointOfInterest.z + s*c3);
							glEnd();
						}
					}
				}
			}
		}
	}


	glFlush();
	glutSwapBuffers();
}

// ----------------------------------------------------------
// Draw function for OpenGL
// ----------------------------------------------------------
void draw() 
{
	drawKinectPointCloud();
}

} // namespace


int main(int argc, char* argv[]) 
{
	
	if (!init(argc, argv)) return 1;
    if (!initKinect()) return 1;
	
	// Initialize webcam
	capture = cvCaptureFromCAM(0); //0=default, -1=any camera, 1..99=your camera
    if(!capture)	return -1;

	// Initialize calibtrator
	//kinectCalibrator();

    // Main loop
    glutMainLoop();

	// Release webcam


    return 0;
}