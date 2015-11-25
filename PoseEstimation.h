#ifndef POSE_ESTIMATION_H
#define POSE_ESTIMATION_H

#include <cstdio>
#include <string>
#include <opencv2/opencv.hpp>

struct Geom {
	cv::Point2d Center;
	cv::Point2d leftEye;
	cv::Point2d rightEye;
	cv::Point2d Nose;
	cv::Point2d Mouth;

	cv::Point2d NoseBase;
	cv::Point2d MidEye;

	double LRdistance;
	double LNdistance;
	double RNdistance;
	double MNdistance;
	double EMdistance;

	double MeanSize;
};

struct Pose {
	double pitch, yaw, roll;
	double slant;
};

void initGeom(cv::Mat_<double> &shape, Geom &G);
void calcPose(Geom &G, Pose &P);
void drawGeom(cv::Mat &img, Geom &G);
void drawPose(cv::Mat &img, Pose &P);

#endif