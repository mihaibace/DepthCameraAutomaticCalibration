#include <Windows.h>
#include <Ole2.h>

#include <gl/GL.h>
#include <gl/GLU.h>
#include <gl/glut.h>

#include <NuiApi.h>
#include <NuiImageCamera.h>
#include <NuiSensor.h>

// ----------------------------------------------------------
// Global Variables
// ----------------------------------------------------------
namespace
{
const int width = 640;
const int height = 480;

double rotate_y=0; 
double rotate_x=0;

// Kinect variables
HANDLE depthStream;				// The indetifier of the Kinect's Depth Camera
INuiSensor* sensor;            // The kinect sensor


// ----------------------------------------------------------
// Function Prototypes
// ----------------------------------------------------------
void draw(void);
void display();
void specialKeys();

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

		default: // Undefined key
			break;
	}
 
	//  Request display update
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
	glLoadIdentity();
 
	// Rotate when user changes rotate_x and rotate_y
	glRotatef( rotate_x, 1.0, 0.0, 0.0 );
	glRotatef( rotate_y, 0.0, 1.0, 0.0 );

	// Scale all the coordinates: for visualisation purposes
	glScalef( 0.2, 0.2, 0.2 );        

	// Get the points data from the Kinect
	USHORT data[width*height];  // array containing the depth information of each pixel
	getKinectData(data);

	// Display the points as a 3D point cloud
	glBegin(GL_POINTS);
		glColor3f(   1.0,  0.0,  0.0 );

		for (int y = 0; y < height; ++y)
		{
			USHORT *line = data + y * width;
			for (int x = 0; x < width; ++x)
			{
				Vector4 pointToDisplay = NuiTransformDepthImageToSkeleton(x,y, line[x]);
				glVertex3f(pointToDisplay.x, pointToDisplay.y, pointToDisplay.z);
			}
		}
	glEnd();

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

    // Main loop
    glutMainLoop();

    return 0;
}