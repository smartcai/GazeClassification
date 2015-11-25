#include "FaceAlignment.h"
using namespace std;
using namespace cv;

Mat_<double> & Tree::operator()(const vector<double> &featPixVals)
{
	int i = 0;
	while (i < feats.size()) {
		double diff = featPixVals[feats[i].idx1] - featPixVals[feats[i].idx2];
		if (diff > feats[i].thresh)
			i = leftChild(i);
		else i = rightChild(i);
	}
	return leafVals[i - feats.size()];
}

void Tree::read(FILE *file)
{
	int nFeat;
	fscanf(file, "%d", &nFeat);
	feats.resize(nFeat);
	for (int i = 0; i < nFeat; i++)
		fscanf(file, "%d%d%lf", &feats[i].idx1, &feats[i].idx2, &feats[i].thresh);
	
	int nLVals;
	fscanf(file, "%d", &nLVals);
	leafVals.resize(nLVals);
	for (int i = 0; i < nLVals; i++) {
		int num;
		fscanf(file, "%d", &num);
		leafVals[i] = Mat::zeros(num, 2, CV_64FC1);
		for (int j = 0; j < num; j++)
			fscanf(file, "%lf%lf", &leafVals[i](j, 0), &leafVals[i](j, 1));
	}
}

void Tree::write(FILE *file)
{
	fprintf(file, "%u\n", feats.size());
	for (int i = 0; i < feats.size(); i++)
		fprintf(file, "%d %d %lf ", feats[i].idx1, feats[i].idx2, feats[i].thresh);
	fprintf(file, "\n");

	fprintf(file, "%u\n", leafVals.size());
	for (int i = 0; i < leafVals.size(); i++) {
		fprintf(file, "%d\n", leafVals[i].rows);
		for (int j = 0; j < leafVals[i].rows; j++)
			fprintf(file, "%lf %lf ", leafVals[i](j, 0), leafVals[i](j, 1));
		fprintf(file, "\n");
	}
	fprintf(file, "\n");
}

ShapePredictor::ShapePredictor(const Mat_<double> &refshape,
							   const vector<vector<Tree> > &randomforests,
							   const vector<Mat_<double> > &pixLocs
							   ) : refShape(refshape), rfs(randomforests)
{
	anchorIdx.resize(pixLocs.size());
	deltas.resize(pixLocs.size());
	// Each cascade uses a different set of pixels for its features.
	// Compute their representations relative to the initial shape.
	for (int i = 0; i < pixLocs.size(); i++)
		createShapeEncoding(refShape, pixLocs[i], anchorIdx[i], deltas[i]);
}

Mat_<double> ShapePredictor::operator()(const Mat_<uchar> &img,
										const BBox &bbox)
{
	Mat_<double> curShape = refShape.clone();
	vector<double> featPixVals;
	
	for (int i = 0; i < rfs.size(); i++) {
		calcFeatPixVals(img, bbox, curShape, refShape,
						anchorIdx[i], deltas[i], featPixVals);

		// evaluate all the trees at this level of the cascade
		for (int j = 0; j < rfs[i].size(); j++) {
			curShape += rfs[i][j](featPixVals);
		}
	}
	curShape = reprojectShape(curShape, bbox);
	return curShape;
}

void ShapePredictor::load(string path)
{
	FILE *file = fopen(path.c_str(), "r");
	if (!file) {
		fprintf(stderr, "Unable to open file\n");
		return;
	}

	int nLdmk;
	fscanf(file, "%d", &nLdmk);
	refShape = Mat::zeros(nLdmk, 2, CV_64FC1);
	for (int i = 0; i < nLdmk; i++)
		fscanf(file, "%lf%lf", &refShape(i, 0), &refShape(i, 1));
	
	int nAnchor;
	fscanf(file, "%d", &nAnchor);
	anchorIdx.resize(nAnchor);
	for (int i = 0; i < nAnchor; i++) {
		int num;
		fscanf(file, "%d", &num);
		anchorIdx[i].resize(num);
		for (int j = 0; j < num; j++)
			fscanf(file, "%d", &anchorIdx[i][j]);
	}
	
	int nDeltas;
	fscanf(file, "%d", &nDeltas);
	deltas.resize(nDeltas);
	for (int i = 0; i < nDeltas; i++) {
		int num;
		fscanf(file, "%d", &num);
		deltas[i] = Mat::zeros(num, 2, CV_64FC1);
		for (int j = 0; j < num; j++)
			fscanf(file, "%lf%lf", &deltas[i](j, 0), &deltas[i](j, 1));
	}
	
	int cascade;
	fscanf(file, "%d", &cascade);
	rfs.resize(cascade);
	for (int i = 0; i < cascade; i++) {
		int num;
		fscanf(file, "%d", &num);
		rfs[i].resize(num);
		for (int j = 0; j < num; j++)
			rfs[i][j].read(file);
	}

	fclose(file);
}

void ShapePredictor::save(string path)
{
	FILE *file = fopen(path.c_str(), "w");
	if (!file) {
		fprintf(stderr, "Unable to create file\n");
		return;
	}

	fprintf(file, "%u\n", refShape.rows);
	for (int i = 0; i < refShape.rows; i++)
		fprintf(file, "%lf %lf ", refShape(i, 0), refShape(i, 1));
	fprintf(file, "\n");

	fprintf(file, "\n%u\n", anchorIdx.size());
	for (int i = 0; i < anchorIdx.size(); i++) {
		fprintf(file, "%u\n", anchorIdx[i].size());
		for (int j = 0; j < anchorIdx[i].size(); j++)
			fprintf(file, "%d ", anchorIdx[i][j]);
		fprintf(file, "\n");
	}

	fprintf(file, "\n%u\n", deltas.size());
	for (int i = 0; i < deltas.size(); i++) {
		fprintf(file, "%d\n", deltas[i].rows);
		for (int j = 0; j < deltas[i].rows; j++)
			fprintf(file, "%lf %lf ", deltas[i](j, 0), deltas[i](j, 1));
		fprintf(file, "\n");
	}

	fprintf(file, "\n%u\n", rfs.size());
	for (int i = 0; i < rfs.size(); i++) {
		fprintf(file, "%u\n", rfs[i].size());
		for (int j = 0; j < rfs[i].size(); j++)
			rfs[i][j].write(file);
		fprintf(file, "\n");
	}

	fclose(file);
}