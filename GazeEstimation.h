#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include "FaceAlignment.h"
#include "PoseEstimation.h"

#define LEFT_EYE 0
#define RIGHT_EYE 1
#define EmptyBox(bbox) (bbox.x == 0 && bbox.y == 0 && bbox.w == 0 && bbox.h == 0)

//class MDA {
//public:
//	cv::Mat_<double> eigenVecs;
//	cv::Mat_<double> eigenVals;
//
//	MDA() {}
//	MDA(cv::Mat_<float> &data, cv::Mat_<float> &labels)
//	{
//		this->compute(data, labels);
//	}
//	void save(const std::string &filename);
//	void load(const std::string &filename);
//
//	cv::Mat_<float> project(cv::Mat_<float> &src);
//	void compute(cv::Mat_<float> &data, cv::Mat_<float> &labels);
//};

cv::Mat_<double> readShape(std::string filename, int ldmkNum);
BBox getTrainBBox(cv::Mat_<uchar> &gray, cv::Mat_<double> &shape, cv::CascadeClassifier &classifier);
BBox getTestBBox(cv::Mat_<uchar> &gray, cv::CascadeClassifier &classifier);

cv::Rect findEyeRect(cv::Mat_<uchar> &gray,
					 cv::Mat_<double> &shape,
					 int whichEye);

std::vector<cv::Point> findEyeCont(cv::Rect &EyeRect,
								   cv::Mat_<double> &shape,
								   int whichEye);

cv::Mat_<uchar> findEyeMask(cv::Mat_<uchar> &EyeGray,
							std::vector<cv::Point> &contour);

void helenTrain();
void helenTest();
void truthCalib();