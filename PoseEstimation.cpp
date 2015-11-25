#include "PoseEstimation.h"
using namespace std;
using namespace cv;

double FindAngle(Point2d &pt1, Point2d &pt2)
{
	double dx = pt2.x - pt1.x;
	double dy = pt2.y - pt1.y;
	double angle = fastAtan2(dy, dx);
	return angle;
}

double FindSlant(int ln, int lf, float Rn, float tita)
{
	float dz = 0;
	float m1 = ((float)ln*ln) / ((float)lf*lf);
	float m2 = (cos(tita))*(cos(tita));

	if (m2 == 1)
		dz = sqrt((Rn*Rn) / (m1 + (Rn*Rn)));
	if (m2 >= 0 && m2 < 1) {
		dz = sqrt(((Rn*Rn) - m1 - 2 * m2*(Rn*Rn)
				   + sqrt(((m1 - (Rn*Rn))*(m1 - (Rn*Rn))) + 4 * m1*m2*(Rn*Rn))
				   ) / (2 * (1 - m2)*(Rn*Rn)));
	}
	double slant = acos(dz);
	return slant;
}

double FindDistance(Point2d &pt1, Point2d &pt2)
{
	double dx = pt2.x - pt1.x;
	double dy = pt2.y - pt1.y;
	return sqrt(dx*dx + dy*dy);
}

void initGeom(Mat_<double> &shape, Geom &G)
{
	// extract face landmarks
	for (int i = 36; i < 42; i++) {
		G.leftEye.x += shape(i, 0);
		G.leftEye.y += shape(i, 1);
	}
	G.leftEye.x /= 6;
	G.leftEye.y /= 6;

	for (int i = 42; i < 48; i++) {
		G.rightEye.x += shape(i, 0);
		G.rightEye.y += shape(i, 1);
	}
	G.rightEye.x /= 6;
	G.rightEye.y /= 6;

	G.MidEye.x = (G.leftEye.x + G.rightEye.x) / 2;
	G.MidEye.y = (G.leftEye.y + G.rightEye.y) / 2;

	G.Nose.x = shape(30, 0);
	G.Nose.y = shape(30, 1);

	/*for (int i = 31; i < 36; i++) {
		G.NoseBase.x += shape(i, 0);
		G.NoseBase.y += shape(i, 1);
	}
	G.NoseBase.x /= 5;
	G.NoseBase.y /= 5;*/


	G.Mouth.x = (shape(51, 0) + shape(62, 0) + shape(66, 0) + shape(57, 0)) / 4;
	G.Mouth.y = (shape(51, 1) + shape(62, 1) + shape(66, 1) + shape(57, 1)) / 4;

	double A = 1.0;
	double B = -atan2(G.MidEye.y - G.Mouth.y, G.MidEye.x - G.Mouth.x);
	double C = -A*G.MidEye.y - B*G.MidEye.x;
	double x0 = G.Nose.x;
	double y0 = G.Nose.y;
	G.NoseBase.x = (B*B*x0 - A*B*y0 - A*C) / (A*A + B*B);
	G.NoseBase.y = (-A*B*x0 - A*A*y0 - B*C) / (A*A + B*B);

	G.LNdistance = FindDistance(G.leftEye, G.Nose);
	G.RNdistance = FindDistance(G.rightEye, G.Nose);
	G.MNdistance = FindDistance(G.Mouth, G.Nose);
	G.EMdistance = FindDistance(G.MidEye, G.Mouth);
	
	G.MeanSize = (G.LNdistance + G.RNdistance + G.MNdistance + G.EMdistance) / 4;
}

void calcPose(Geom &G, Pose &P)
{
	float imageFacialNormalLength = FindDistance(G.Nose, G.NoseBase);

	P.roll = FindAngle(G.leftEye, G.rightEye);
	P.roll = (P.roll < 180) ? P.roll : P.roll - 360;

	double symm = FindAngle(G.NoseBase, G.MidEye);
	double tilt = FindAngle(G.NoseBase, G.Nose);
	double tita = abs(tilt - symm)*(3.1416 / 180);

	P.slant = FindSlant(imageFacialNormalLength, G.EMdistance, 0.5, tita);

	Point3d normal;	// define a 3D vector for facial normal
	normal.x = sin(P.slant)*cos((360 - tilt)*(3.1416 / 180));
	normal.y = sin(P.slant)*sin((360 - tilt)*(3.1416 / 180));
	normal.z = -cos(P.slant);

	P.pitch = acos(sqrt((normal.x*normal.x + normal.z*normal.z)
						/ (normal.x*normal.x + normal.y*normal.y + normal.z*normal.z)));
	if (G.Nose.y < G.NoseBase.y) P.pitch = -P.pitch;
	P.pitch *= 180 / 3.1416;


	P.yaw = acos((abs(normal.z)) / sqrt(normal.x*normal.x + normal.z*normal.z));
	if (G.Nose.x < G.NoseBase.x) P.yaw = -P.yaw;
	P.yaw *= 180 / 3.1416;
}

void drawGeom(Mat &img, Geom &G)
{
	line(img, G.leftEye, G.rightEye, CV_RGB(0, 0, 255));
	line(img, G.leftEye, G.Nose, CV_RGB(0, 0, 255));
	line(img, G.rightEye, G.Nose, CV_RGB(0, 0, 255));
	line(img, G.leftEye, G.Mouth, CV_RGB(0, 0, 255));
	line(img, G.rightEye, G.Mouth, CV_RGB(0, 0, 255));
	line(img, G.Nose, G.Mouth, CV_RGB(0, 0, 255));
	line(img, G.MidEye, G.Mouth, CV_RGB(0, 0, 255));
	line(img, G.Nose, G.NoseBase, CV_RGB(0, 0, 255));
}

void drawPose(Mat &img, Pose &P)
{

}