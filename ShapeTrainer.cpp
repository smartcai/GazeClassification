#include "FaceAlignment.h"
using namespace std;
using namespace cv;

Tree ShapeTrainer::buildTree(vector<TrainSample> &samples,
							 const Mat_<double> &pixLocs)
{
	deque<pair<int, int> > parts;
	parts.push_back(make_pair(0, (int)samples.size()));

	Tree tree;

	// walk the tree in breadth first order
	const int nSplitFeats = (int)pow(2, treeDepth - 1);
	vector<Mat_<double> > sums(nSplitFeats * 2 + 1);
	for (int i = 0; i < sums.size(); i++)
		sums[i] = Mat::zeros(nLdmk, 2, CV_64FC1);
	for (int i = 0; i < samples.size(); i++) {
		sums[0] += samples[i].tarShape - samples[i].curShape;
	}

	for (int i = 0; i < nSplitFeats; i++) {
		pair<int, int> range = parts.front();
		parts.pop_front();

		Feat feat = splitFeat(samples, range.first, range.second, pixLocs,
							  sums[i], sums[leftChild(i)], sums[rightChild(i)]);
		tree.feats.push_back(feat);

		int mid = divideSamples(feat, samples, range.first, range.second);
		parts.push_back(make_pair(range.first, mid));
		parts.push_back(make_pair(mid, range.second));
	}

	// Now all the parts contain the ranges for the leaves so
	// we can use them to compute the average leaf values.
	tree.leafVals.resize(parts.size());
	for (int i = 0; i < parts.size(); i++) {
		if (parts[i].first != parts[i].second)
			tree.leafVals[i] = sums[nSplitFeats + i] * nu / (parts[i].second - parts[i].first);
		else
			tree.leafVals[i] = Mat::zeros(nLdmk, 2, CV_64FC1);

		/*if (abs(tree.leafVals[i](0, 0)) > 1) {
		printf("\nsums[nsf+%d]: ", i);
		for (int j = 0; j < 5; j++)
		cout << sums[nSplitFeats + i](j, 0) << " ";
		cout << endl;
		exit(0);
		}*/

		// adjust current shape based on these predictions
		for (int j = parts[i].first; j < parts[i].second; j++) {
			samples[j].curShape += tree.leafVals[i];
		}
	}

	return tree;
}

Feat ShapeTrainer::randFeat(const Mat_<double> & pixLocs)
{
	RNG rng(getTickCount());

	Feat feat;
	double prob;
	do {
		feat.idx1 = rng.uniform(0, featPoolSize);
		feat.idx2 = rng.uniform(0, featPoolSize);
		double dx = pixLocs(feat.idx1, 0) - pixLocs(feat.idx2, 0);
		double dy = pixLocs(feat.idx1, 1) - pixLocs(feat.idx2, 1);
		double dist = sqrt(dx*dx + dy*dy);
		prob = exp(-dist / lambda);
	} while (feat.idx1 == feat.idx2 || prob < rng.uniform(0.0, 1.0));

	feat.thresh = (rng.uniform(0.0, 1.0) * 256 - 128) / 2.0;
	return feat;
}

Feat ShapeTrainer::splitFeat(const vector<TrainSample> &samples,
							 const int begin,
							 const int end,
							 const Mat_<double> &pixLocs,
							 Mat_<double> &sum,
							 Mat_<double> &lsum,
							 Mat_<double> &rsum)
{
	// sample the random features we test here
	vector<Feat> feats;
	feats.reserve(nTestFeats);
	for (int i = 0; i < nTestFeats; i++)
		feats.push_back(randFeat(pixLocs));

	vector<Mat_<double> > lsums(nTestFeats);
	for (int i = 0; i < nTestFeats; i++)
		lsums[i] = Mat::zeros(nLdmk, 2, CV_64FC1);
	vector<int> lcnt(nTestFeats, 0);
	// calculate sums of vectors which go left for each feat
	for (int j = begin; j < end; j++) {
		Mat_<double> tmp = samples[j].tarShape - samples[j].curShape;
		for (int i = 0; i < nTestFeats; i++) {
			double val1 = samples[j].featPixVals[feats[i].idx1];
			double val2 = samples[j].featPixVals[feats[i].idx2];
			if (val1 - val2 > feats[i].thresh) {
				lsums[i] += tmp;
				lcnt[i] += 1;
			}
		}
	}

	// now figure out which feature is the best
	double maxscore = -1;
	int maxidx = 0;
	for (int i = 0; i < nTestFeats; i++) {
		double score = 0;
		int rcnt = end - begin - lcnt[i];
		if (lcnt[i] != 0 && rcnt != 0) {
			Mat_<double> tmp = sum - lsums[i];
			score = lsums[i].dot(lsums[i]) / lcnt[i] + tmp.dot(tmp) / rcnt;
			if (score > maxscore) {
				maxscore = score;
				maxidx = i;
			}
		}
	}

	lsum = lsums[maxidx];
	rsum = sum - lsum;

	return feats[maxidx];
}

int ShapeTrainer::divideSamples(const Feat &feat,
								vector<TrainSample> &samples,
								int begin,
								int end)
{
	// splits samples based on split (sorta like in quick sort)
	// and returns the mid point. make sure you return the mid
	// in a way compatible with how we walk through the tree.
	int i = begin;
	for (int j = begin; j < end; j++) {
		double val1 = samples[j].featPixVals[feat.idx1];
		double val2 = samples[j].featPixVals[feat.idx2];
		if (val1 - val2 > feat.thresh) {
			samples[i].swap(samples[j]);
			i++;
		}
	}
	return i;
}

Mat_<double> ShapeTrainer::populateSamples(const vector<Mat_<double> > &shapes,
										   const vector<BBox> &bboxes,
										   vector<TrainSample> &samples)
{
	samples.clear();
	Mat_<double> meanShape = Mat::zeros(shapes[0].rows, 2, CV_64FC1);

	// fill out target shapes
	RNG rng(getTickCount());
	for (int i = 0; i < shapes.size(); i++) {
		TrainSample samp;
		samp.imgIdx = i;
		samp.bbox = bboxes[i];
		samp.tarShape = projectShape(shapes[i], bboxes[i]);
		for (int j = 0; j < nOversampling; j++) {
			samples.push_back(samp);
		}
		meanShape += samp.tarShape;
	}
	meanShape /= shapes.size();

	// pick random initial shapes
	for (int i = 0; i < samples.size(); i++) {
		if (i % nOversampling == 0) {
			// The mean shape is what we really use as an initial shape so always
			// include it in the training set as an example starting shape.
			samples[i].curShape = meanShape;
		}
		else {
			// Pick a random convex combination of two of the target shapes and use
			// that as the initial shape for this sample.
			RNG rng(getTickCount());
			int randidx1 = rng.uniform(0, samples.size());
			int randidx2 = rng.uniform(0, samples.size());
			double alpha = rng.uniform(0.0, 1.0);
			samples[i].curShape = alpha*samples[randidx1].tarShape +
				(1 - alpha)*samples[randidx2].tarShape;
		}
	}

	return meanShape;
}

void ShapeTrainer::randPixLoc(Mat_<double> &pixLocs,
							  const double minx,
							  const double miny,
							  const double maxx,
							  const double maxy)
{
	RNG rng(getTickCount());
	pixLocs = Mat::zeros(featPoolSize, 2, CV_64FC1);
	for (int i = 0; i < featPoolSize; i++) {
		pixLocs(i, 0) = rng.uniform(0.0, 1.0)*(maxx - minx) + minx;
		pixLocs(i, 1) = rng.uniform(0.0, 1.0)*(maxy - miny) + miny;
	}
}

vector<Mat_<double> > ShapeTrainer::samplePixLocs(const Mat_<double> &refShape)
{
	double minx, maxx, miny, maxy;
	minMaxLoc(refShape.col(0), &minx, &maxx);
	minMaxLoc(refShape.col(1), &miny, &maxy);

	vector<Mat_<double> > pixLocs;
	pixLocs.resize(cascadeDepth);
	for (int i = 0; i < cascadeDepth; i++)
		randPixLoc(pixLocs[i], minx, miny, maxx, maxy);
	return pixLocs;
}

ShapePredictor ShapeTrainer::train(const vector<Mat_<uchar> > &images,
								   const vector<Mat_<double> > &shapes,
								   const vector<BBox> &bboxes)
{
	nLdmk = shapes[0].rows;
	vector<TrainSample> samples;
	Mat_<double> refShape = populateSamples(shapes, bboxes, samples);
	vector<Mat_<double> > pixLocs = samplePixLocs(refShape);

	vector<vector<Tree> > rfs(cascadeDepth);
	for (int cascade = 0; cascade < cascadeDepth; cascade++) {
		cout << "------------------------------------------------------------" << endl;
		cout << "training cascade " << cascade + 1 << endl;
		// Each cascade uses a different set of pixels for its features.  We compute
		// their representations relative to the initial shape first.
		vector<int> anchorIdx;
		Mat_<double> deltas;
		createShapeEncoding(refShape, pixLocs[cascade], anchorIdx, deltas);

		// First compute the feature pixel values for each 
		// training sample at this level of the cascade.
		for (int i = 0; i < samples.size(); i++) {
			calcFeatPixVals(images[samples[i].imgIdx], samples[i].bbox,
							samples[i].curShape, refShape, anchorIdx,
							deltas, samples[i].featPixVals);
		}

		// Now start building the trees at this cascade level.
		int dotstep = treesPerLevel / 50;
		for (int i = 0; i < treesPerLevel; i++) {
			rfs[cascade].push_back(buildTree(samples, pixLocs[cascade]));
			if (i % dotstep == 0) cout << '.';
		}
		cout << "done" << endl;
	}
	cout << "------------------------------------------------------------" << endl;
	cout << "Building ShapePredictor..." << endl;
	return ShapePredictor(refShape, rfs, pixLocs);
}