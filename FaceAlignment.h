#ifndef FACE_ALIGNMENT_H
#define FACE_ALIGNMENT_H

#include <cstdio>
#include <string>
#include <vector>
#include <deque>
#include <utility>
#include <opencv2/opencv.hpp>

class BBox {
public:
	int x, y;
	int w, h;
	int cx, cy;
	BBox() { }
	BBox(int _x, int _y, int _w, int _h) 
	{
		x = _x;
		y = _y;
		w = _w;
		h = _h;
		cx = x + w / 2.0;
		cy = y + h / 2.0;
	}
	BBox(cv::Rect rect)
	{
		x = rect.x;
		y = rect.y;
		w = rect.width;
		h = rect.height;
		cx = x + w / 2.0;
		cy = y + h / 2.0;
	}
};

struct Feat {
	int idx1;
	int idx2;
	double thresh;
};

class Tree {
public:
	std::vector<Feat> feats;
	std::vector<cv::Mat_<double> > leafVals;

	cv::Mat_<double> & operator()(const std::vector<double> &featPixVals);

	void read(FILE *file);

	void write(FILE *file);
};

class ShapePredictor {
private:
	cv::Mat_<double> refShape;
	std::vector<std::vector<Tree> > rfs;
	std::vector<std::vector<int> > anchorIdx;
	std::vector<cv::Mat_<double> > deltas;

public:
	ShapePredictor() { }

	ShapePredictor(const cv::Mat_<double> &refshape,
				   const std::vector<std::vector<Tree> > &randomforests,
				   const std::vector<cv::Mat_<double> > &pixLocs);

	cv::Mat_<double> operator()(const cv::Mat_<uchar> &img, const BBox &bbox);

	void load(std::string path);

	void save(std::string path);
};

class ShapeTrainer {
private:
	int nLdmk;
	int treeDepth;
	int cascadeDepth;
	int treesPerLevel;
	int nOversampling;
	int featPoolSize;
	int nTestFeats;
	double nu;
	double lambda;

	struct TrainSample {
		int imgIdx;
		BBox bbox;
		cv::Mat_<double> tarShape;	// target shape
		cv::Mat_<double> curShape;	// current shape
		std::vector<double> featPixVals;

		void swap(TrainSample &item)
		{
			std::swap(imgIdx, item.imgIdx);
			std::swap(bbox.x, item.bbox.x);
			std::swap(bbox.y, item.bbox.y);
			std::swap(bbox.w, item.bbox.w);
			std::swap(bbox.h, item.bbox.h);
			std::swap(bbox.cx, item.bbox.cx);
			std::swap(bbox.cy, item.bbox.cy);

			cv::Mat_<double> tmp;

			tmp = tarShape.clone();
			tarShape = item.tarShape.clone();
			item.tarShape = tmp.clone();

			tmp = curShape.clone();
			curShape = item.curShape.clone();
			item.curShape = tmp.clone();

			featPixVals.swap(item.featPixVals);
		}
	};

	Tree buildTree(std::vector<TrainSample> &samples,
				   const cv::Mat_<double> &pixLocs);

	Feat randFeat(const cv::Mat_<double> & pixLocs);

	Feat splitFeat(const std::vector<TrainSample> &samples,
				   const int begin,
				   const int end,
				   const cv::Mat_<double> &pixLocs,
				   cv::Mat_<double> &sum,
				   cv::Mat_<double> &lsum,
				   cv::Mat_<double> &rsum);

	int divideSamples(const Feat &feat,
					  std::vector<TrainSample> &samples,
					  int begin,
					  int end);

	cv::Mat_<double> populateSamples(const std::vector<cv::Mat_<double> > &shapes,
									 const std::vector<BBox> &bboxes,
									 std::vector<TrainSample> &samples);

	void randPixLoc(cv::Mat_<double> &pixLocs,
					const double minx,
					const double miny,
					const double maxx,
					const double maxy);

	std::vector<cv::Mat_<double> > samplePixLocs(const cv::Mat_<double> &refShape);

public:
	ShapeTrainer()
	{
		nLdmk = 0;
		treeDepth = 5;
		cascadeDepth = 10;
		treesPerLevel = 500;
		nOversampling = 20;
		featPoolSize = 400;
		nTestFeats = 20;
		nu = 0.1;
		lambda = 0.1;
	}

	void setTreeDepth(int depth = 5) { treeDepth = depth; }
	void setCascadeDepth(int depth = 10) { cascadeDepth = depth; }
	void setTreesPerLevel(int num = 500) { treesPerLevel = num; }
	void setInitSetNum(int num = 20) { nOversampling = num; }
	void setFeatPoolSize(int num = 400) { featPoolSize = num; }
	void setTestFeatNum(int num = 20) { nTestFeats = num; }
	void setLearningRate(double rate = 0.1) { nu = rate; }
	void setLambda(double l = 0.1) { lambda = l; }

	ShapePredictor train(const std::vector<cv::Mat_<uchar> > &images,
						 const std::vector<cv::Mat_<double> > &shapes,
						 const std::vector<BBox> &bboxes);
};


#define leftChild(idx) (2 * idx + 1)
#define rightChild(idx) (2 * idx + 2)

int nearestShapePoint(const cv::Mat_<double> &shape, const cv::Point2f &pt);

void createShapeEncoding(const cv::Mat_<double> &shape,
						 const cv::Mat_<double> &candPix,
						 std::vector<int> &anchorIdx,
						 cv::Mat_<double> &deltas);

void similarityTransform(const cv::Mat_<double> &fromShape,
						 const cv::Mat_<double> &toShape,
						 cv::Mat_<double> &rotation, double &scale);

cv::Mat_<double> projectShape(const cv::Mat_<double> &shape, const BBox &bbox);

cv::Mat_<double> reprojectShape(const cv::Mat_<double> &shape, const BBox &bbox);

void calcFeatPixVals(const cv::Mat_<uchar> &img,
					 const BBox &bbox,
					 const cv::Mat_<double> &curShape,	// normalized shape
					 const cv::Mat_<double> &refShape,	// normalized shape
					 const std::vector<int> &refAnchorIdx,
					 const cv::Mat_<double> &refPixDeltas,
					 std::vector<double> &featPixVals);

void drawShape(cv::Mat &img, cv::Mat_<double> &shape, int thickness);

#endif