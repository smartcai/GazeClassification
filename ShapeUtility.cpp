#include "FaceAlignment.h"
using namespace std;
using namespace cv;

int nearestShapePoint(const Mat_<double> &shape,
					  const Point2f &pt)
{
	// find the nearest part of the shape to this pixel
	int minidx = 0;
	double minsqdist = numeric_limits<double>::infinity();
	for (int i = 0; i < shape.rows; i++) {
		double dx = shape(i, 0) - pt.x;
		double dy = shape(i, 1) - pt.y;
		double sqdist = dx*dx + dy*dy;
		if (sqdist < minsqdist) {
			minsqdist = sqdist;
			minidx = i;
		}
	}
	return minidx;
}

void createShapeEncoding(const Mat_<double> &shape,
						 const Mat_<double> &candPix,
						 vector<int> &anchorIdx,
						 Mat_<double> &deltas)
{
	anchorIdx.resize(candPix.rows);
	deltas = Mat::zeros(candPix.rows, 2, CV_64FC1);

	for (int i = 0; i < candPix.rows; i++) {
		anchorIdx[i] = nearestShapePoint(shape, Point2f(candPix(i, 0), candPix(i, 1)));
		deltas(i, 0) = candPix(i, 0) - shape(anchorIdx[i], 0);
		deltas(i, 1) = candPix(i, 1) - shape(anchorIdx[i], 1);
	}
}

void similarityTransform(const Mat_<double> &fromShape,
						 const Mat_<double> &toShape,
						 Mat_<double> &rotation, double &scale)
{
	rotation = Mat::zeros(2, 2, CV_64FC1);
	scale = 0;

	assert(fromShape.rows == toShape.rows);
	int rows = fromShape.rows;

	double fcx = 0, fcy = 0, tcx = 0, tcy = 0;
	for (int i = 0; i < rows; i++) {
		fcx += fromShape(i, 0);
		fcy += fromShape(i, 1);
		tcx += toShape(i, 0);
		tcy += toShape(i, 1);
	}
	fcx /= rows;
	fcy /= rows;
	tcx /= rows;
	tcy /= rows;

	Mat_<double> ftmp = fromShape.clone();
	Mat_<double> ttmp = toShape.clone();
	for (int i = 0; i < rows; i++) {
		ftmp(i, 0) -= fcx;
		ftmp(i, 1) -= fcy;
		ttmp(i, 0) -= tcx;
		ttmp(i, 1) -= tcy;
	}

	Mat_<double> fcov, tcov;
	Mat_<double> fmean, tmean;
	calcCovarMatrix(ftmp, fcov, fmean, CV_COVAR_COLS);
	calcCovarMatrix(ttmp, tcov, tmean, CV_COVAR_COLS);

	double fsize = sqrtf(norm(fcov));
	double tsize = sqrtf(norm(tcov));
	scale = tsize / fsize;
	ftmp /= fsize;
	ttmp /= tsize;

	double num = 0, den = 0;
	// an efficient way to calculate rotation, using cross and dot production
	for (int i = 0; i < rows; i++) {
		num += ftmp(i, 1)*ttmp(i, 0) - ftmp(i, 0)*ttmp(i, 1);
		den += ftmp(i, 0)*ttmp(i, 0) + ftmp(i, 1)*ttmp(i, 1);
	}
	double norm = sqrtf(num*num + den*den);
	// theta is clock-wise rotation angle(fromShape -> toShape)
	double sinTheta = num / norm;
	double cosTheta = den / norm;

	rotation(0, 0) = cosTheta;
	rotation(0, 1) = sinTheta;
	rotation(1, 0) = -sinTheta;
	rotation(1, 1) = cosTheta;
}

Mat_<double> projectShape(const Mat_<double> &shape, const BBox &bbox)
{
	Mat_<double> result(shape.rows, 2);

	for (int i = 0; i < shape.rows; i++) {
		result(i, 0) = (shape(i, 0) - bbox.cx) / (bbox.w / 2.0);
		result(i, 1) = (shape(i, 1) - bbox.cy) / (bbox.h / 2.0);
	}

	return result;
}

Mat_<double> reprojectShape(const Mat_<double> &shape, const BBox &bbox)
{
	Mat_<double> result(shape.rows, 2);

	for (int i = 0; i < shape.rows; i++) {
		result(i, 0) = bbox.cx + shape(i, 0) * bbox.w / 2.0;
		result(i, 1) = bbox.cy + shape(i, 1) * bbox.h / 2.0;
	}

	return result;
}

void calcFeatPixVals(const Mat_<uchar> &img,
					 const BBox &bbox,
					 const Mat_<double> &curShape,	// normalized shape
					 const Mat_<double> &refShape,	// normalized shape
					 const vector<int> &refAnchorIdx,
					 const Mat_<double> &refPixDeltas,
					 vector<double> &featPixVals)
{
	Mat_<double> rotation;
	double scale;
	similarityTransform(refShape, curShape, rotation, scale);

	featPixVals.resize(refPixDeltas.rows);
	for (int i = 0; i < featPixVals.size(); i++) {
		double projx = rotation(0, 0)*refPixDeltas(i, 0) + rotation(0, 1)*refPixDeltas(i, 1);
		double projy = rotation(1, 0)*refPixDeltas(i, 0) + rotation(1, 1)*refPixDeltas(i, 1);
		projx *= scale;
		projy *= scale;

		int idx = refAnchorIdx[i];
		double realx = bbox.cx + (projx + curShape(idx, 0)) * bbox.w / 2.0;
		double realy = bbox.cy + (projy + curShape(idx, 1)) * bbox.h / 2.0;

		if (0 <= realx && realx < img.cols &&
			0 <= realy && realy < img.rows)
			featPixVals[i] = (double)img(realy, realx);
		else featPixVals[i] = 0;
	}
}

void drawShape(Mat &img, Mat_<double> &shape, int thickness = 1)
{
	Point pt1, pt2;
	// chin contour
	/*for (int i = 0; i < 16; i++) {
		pt1 = Point(shape(i, 0), shape(i, 1));
		pt2 = Point(shape(i + 1, 0), shape(i + 1, 1));
		line(img, pt1, pt2, CV_RGB(0, 255, 0), thickness);
	}*/
	// left eyebrow
	for (int i = 17; i < 21; i++) {
		pt1 = Point(shape(i, 0), shape(i, 1));
		pt2 = Point(shape(i + 1, 0), shape(i + 1, 1));
		line(img, pt1, pt2, CV_RGB(0, 255, 0), thickness);
	}
	// right eyebrow
	for (int i = 22; i < 27 - 1; i++) {
		pt1 = Point(shape(i, 0), shape(i, 1));
		pt2 = Point(shape(i + 1, 0), shape(i + 1, 1));
		line(img, pt1, pt2, CV_RGB(0, 255, 0), thickness);
	}
	// nose contour
	for (int i = 27; i < 35; i++) {
		pt1 = Point(shape(i, 0), shape(i, 1));
		pt2 = Point(shape(i + 1, 0), shape(i + 1, 1));
		line(img, pt1, pt2, CV_RGB(0, 255, 0), thickness);
	}
	pt1 = Point(shape(30, 0), shape(30, 1));
	pt2 = Point(shape(35, 0), shape(35, 1));
	line(img, pt1, pt2, CV_RGB(0, 255, 0), thickness);
	// left eye
	for (int i = 36; i < 41; i++) {
		pt1 = Point(shape(i, 0), shape(i, 1));
		pt2 = Point(shape(i + 1, 0), shape(i + 1, 1));
		line(img, pt1, pt2, CV_RGB(0, 255, 0), thickness);
	}
	pt1 = Point(shape(36, 0), shape(36, 1));
	pt2 = Point(shape(41, 0), shape(41, 1));
	line(img, pt1, pt2, CV_RGB(0, 255, 0), thickness);
	// right eye
	for (int i = 42; i < 47; i++) {
		pt1 = Point(shape(i, 0), shape(i, 1));
		pt2 = Point(shape(i + 1, 0), shape(i + 1, 1));
		line(img, pt1, pt2, CV_RGB(0, 255, 0), thickness);
	}
	pt1 = Point(shape(42, 0), shape(42, 1));
	pt2 = Point(shape(47, 0), shape(47, 1));
	line(img, pt1, pt2, CV_RGB(0, 255, 0), thickness);
	// outer lips
	for (int i = 48; i < 59; i++) {
		pt1 = Point(shape(i, 0), shape(i, 1));
		pt2 = Point(shape(i + 1, 0), shape(i + 1, 1));
		line(img, pt1, pt2, CV_RGB(0, 255, 0), thickness);
	}
	pt1 = Point(shape(48, 0), shape(48, 1));
	pt2 = Point(shape(59, 0), shape(59, 1));
	line(img, pt1, pt2, CV_RGB(0, 255, 0), thickness);
	// inner lips
	for (int i = 60; i < 67; i++) {
		pt1 = Point(shape(i, 0), shape(i, 1));
		pt2 = Point(shape(i + 1, 0), shape(i + 1, 1));
		line(img, pt1, pt2, CV_RGB(0, 255, 0), thickness);
	}
	pt1 = Point(shape(60, 0), shape(60, 1));
	pt2 = Point(shape(67, 0), shape(67, 1));
	line(img, pt1, pt2, CV_RGB(0, 255, 0), thickness);
}