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

// ----------------------------------------------------------
// Global Variables
// ----------------------------------------------------------
namespace
{
const int width = 640;
const int height = 480;

double toggleNormal = 0;
double rotate_y = 0; 
double rotate_x = 0;
double trans_x = 0;
double trans_y = 0;
double trans_z = 0;

// Kinect variables
HANDLE depthStream;				// The indetifier of the Kinect's Depth Camera
INuiSensor* sensor;            // The kinect sensor


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

    return sensor;
}

// ----------------------------------------------------------
// Get data from the Kinect
// ----------------------------------------------------------
void getKinectData(USHORT * dest) 
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

// ----------------------------------------------------------
// Draw Kinect data
// ----------------------------------------------------------
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
	glScalef(0.2, 0.2, 0.2);       

	// Get the points data from the Kinect
	USHORT data[width*height];  // array containing the depth information of each pixel
	getKinectData(data);

	// Display the points as a 3D point cloud
	glBegin(GL_POINTS);
		glColor3f(1.0, 0.0, 0.0);

		for (int y = 0; y < height; ++y)
		{
			USHORT *line = data + y * width;
			for (int x = 0; x < width; ++x)
			{
				Vector4 pointToDisplay = NuiTransformDepthImageToSkeleton(x, y, line[x]);
				glVertex3f(pointToDisplay.x, pointToDisplay.y, pointToDisplay.z);
			}
		}
	glEnd();

	// Toggle variable that allows to have a button that controls the display of the normal vectors
	if (toggleNormal == 1)
	{
		// Normal estimation: try to find 4 connected neighbours and estimate the normal based on them
		for (int y = 0; y < height; y+=4)
		{
			USHORT *line = data + y * width;
			for (int x = 0; x < width; x+=4)
			{
				// Verify that all the neighbours are within bounds
				if ((x-1 >= 0) && (x+1 <= width) && (y-1 >= 0) && (y+1 <= height))
				{
					Vector4 pointOfInterest = NuiTransformDepthImageToSkeleton(x, y, line[x]);
					Vector4 nLeft = NuiTransformDepthImageToSkeleton(x-1, y, line[x-1]);
					Vector4 nRight = NuiTransformDepthImageToSkeleton(x+1, y, line[x+1]);
					Vector4 nUp = NuiTransformDepthImageToSkeleton(x, y-1, line[x - width]);
					Vector4 nDown = NuiTransformDepthImageToSkeleton(x, y+1, line[x + width]);

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
	glPushMatrix();
	glScalef(1000, 1000, 1000);
	glBegin(GL_LINES);
	glColor3f(1,0,0);
	glVertex3f(0,0,0);
	glVertex3f(1,0,0);
	glColor3f(0,1,0);
	glVertex3f(0,0,0);
	glVertex3f(0,1,0);
	glColor3f(0,0,1);
	glVertex3f(0,0,0);
	glVertex3f(0,0,1);
	glEnd();
	glPopMatrix();
}

} // namespace

int main(int argc, char* argv[]) 
{
	
	if (!init(argc, argv)) return 1;
    if (!initKinect()) return 1;

    // Main loop
    glutMainLoop();

    return 0;
}