#include "FaceAlignment.h"
#include "GazeEstimation.h"
#include "PoseEstimation.h"
using namespace std;
using namespace cv;

void calcMultiHog(Mat_<uchar> &img, vector<float> &multihog)
{
	HOGDescriptor des1(Size(16, 16), Size(8, 8), Size(4, 4), Size(8, 8), 12, 1, -1, NORM_L2);
	HOGDescriptor des2(Size(8, 8), Size(4, 4), Size(2, 2), Size(4, 4), 12, 1, -1, NORM_L2);
	vector<float> hog1, hog2;
	des1.compute(img, hog1, Size(16, 16));
	des2.compute(img, hog2, Size(8, 8));
	normalize(hog1, hog1, 1.0, NORM_MINMAX);
	normalize(hog2, hog2, 1.0, NORM_MINMAX);
	for (int i = 0; i < hog1.size(); i++)
		multihog.push_back(hog1[i]);
	for (int i = 0; i < hog2.size(); i++)
		multihog.push_back(hog2[i]);
}

void regularize(Mat_<uchar> &gray,
				BBox &bbox, Pose &P,
				Mat_<double> &shape,
				Mat_<uchar> &lEye,
				Mat_<uchar> &rEye)
{
	double w = bbox.w;
	double h = bbox.h;
	double scale = 1.5;
	double minx = max(0.0, bbox.x - (scale - 1)*w / 2);
	double miny = max(0.0, bbox.y - (scale - 1)*h / 2);
	double maxx = min(gray.cols - 1.0, minx + scale*w);
	double maxy = min(gray.rows - 1.0, miny + scale*h);
	w = maxx - minx;
	h = maxy - miny;

	BBox oldbox(minx, miny, w, h);
	BBox newbox(0, 0, w, h);

	Mat_<uchar> faceImg = gray(Rect(minx, miny, w, h));
	Mat_<uchar> regFace = Mat_<uchar>::zeros(faceImg.size());
	Mat rotMat = getRotationMatrix2D(Point2f(w / 2.0, h / 2.0), P.roll, 1.0);
	warpAffine(faceImg, regFace, rotMat, faceImg.size());

	double cosa = rotMat.at<double>(0, 0);
	double sina = rotMat.at<double>(0, 1);
	Mat_<double> projShape = projectShape(shape, oldbox);
	Mat_<double> newShape = Mat_<double>::zeros(shape.size());
	for (int i = 0; i < shape.rows; i++) {
		newShape(i, 0) = cosa*projShape(i, 0) + sina*projShape(i, 1);
		newShape(i, 1) = -sina*projShape(i, 0) + cosa*projShape(i, 1);
	}
	newShape = reprojectShape(newShape, newbox);

	// get eye images & data
	Rect lRect = findEyeRect(regFace, newShape, LEFT_EYE);
	Rect rRect = findEyeRect(regFace, newShape, RIGHT_EYE);
	vector<Point> lCont = findEyeCont(lRect, newShape, LEFT_EYE);
	vector<Point> rCont = findEyeCont(rRect, newShape, RIGHT_EYE);
	lEye = regFace(lRect);
	rEye = regFace(rRect);
	// set mask and crop
	Mat_<uchar> lMask = findEyeMask(lEye, lCont);
	Mat_<uchar> rMask = findEyeMask(rEye, rCont);
	//lEye.setTo(0, ~lMask);
	//rEye.setTo(0, ~rMask);
	normalize(lEye, lEye, 255, 0, NORM_MINMAX);
	normalize(rEye, rEye, 255, 0, NORM_MINMAX);
	// resize to proper size
	resize(lEye, lEye, Size(64, 32));
	resize(rEye, rEye, Size(64, 32));
	
}

void collectData(int subjId,
				 CascadeClassifier &classifier,
				 ShapePredictor &predictor,
				 Mat_<float> &labels,
				 Mat_<float> &multihog,
				 Mat_<float> &landmarks)
{
	int H[] = { -15, -10, -5, 0, 5, 10, 15 };
	int V[] = { -10, 0, 10 };

	string path = to_string(subjId) + "/";
	if (subjId < 10) path = "columbia/000" + path;
	else path = "columbia/00" + path;
	ifstream fin(path + "annotation.txt");

	for (int imgId = 0; imgId < 105; imgId++) {
		int p, v, h;
		fin >> p >> v >> h;
		if (abs(p) > 15) continue;
		string imgpath = path + to_string(imgId) + ".jpg";
		Mat_<uchar> img = imread(imgpath, 0);
		BBox bbox = getTestBBox(img, classifier);
		if (EmptyBox(bbox)) continue;

		int l = 0;
		// EYE, MOUTH, NOF
		if (abs(h) <= 5 && v == 0) l = 0;
		else if (abs(h) <= 5 && v == -10) l = 1;
		else l = 2;

		if (l == 2) {
			RNG rng(getTickCount());
			double num = rng.uniform(0.0, 1.0);
			if (num > 0.5) continue;
		}

		// 上中下
		/*if (v < 0) l = 0;
		else if (v == 0) l = 1;
		else l = 2;*/

		// 9分类
		/*if (h < -5) l += 0;
		else if (h > 5) l += 2;
		else l += 1;
		if (v < 0) l += 0;
		else if (v > 0) l += 2 * 3;
		else l += 1 * 3;*/

		Mat_<float> lab = l*Mat_<float>::ones(1, 1);
		labels.push_back(lab);

		Mat_<double> shape = predictor(img, bbox);
		Geom G;	initGeom(shape, G);
		Pose P; calcPose(G, P);

		Mat_<uchar> lEye, rEye;
		regularize(img, bbox, P, shape, lEye, rEye);

		vector<float> lRlt;
		vector<float> rRlt;
		calcMultiHog(lEye, lRlt);
		calcMultiHog(rEye, rRlt);

		vector<float> _hog2nd_vec;
		for (int k = 0; k < lRlt.size(); k++)
			_hog2nd_vec.push_back(lRlt[k]);
		for (int k = 0; k < rRlt.size(); k++)
			_hog2nd_vec.push_back(rRlt[k]);
		Mat_<float> _hog2nd_row = Mat_<float>(_hog2nd_vec).reshape(1, 1);
		multihog.push_back(_hog2nd_row);

		vector<float> _ldmks;
		for (int i = 28; i < 48; i++) {
			_ldmks.push_back((shape(i, 0) - bbox.cx) / bbox.w);
			_ldmks.push_back((shape(i, 1) - bbox.cy) / bbox.h);
		}
		float mouthx = (shape(51, 0) + shape(62, 0) + shape(66, 0) + shape(57, 0)) / 4;
		float mouthy = (shape(51, 1) + shape(62, 1) + shape(66, 1) + shape(57, 1)) / 4;
		_ldmks.push_back((mouthx - bbox.cx) / bbox.w);
		_ldmks.push_back((mouthy - bbox.cy) / bbox.h);
		float maxVal = *std::max_element(_ldmks.begin(), _ldmks.end());
		for (int i = 0; i < _ldmks.size(); i++) _ldmks[i] *= 1.0 / maxVal; // scale to [-1, 1]

		Mat_<float> ldmks = Mat_<float>(_ldmks).reshape(1, 1);
		landmarks.push_back(ldmks);
	}
	fin.close();
}

void columbiaTrain(int testId = 0)
{
	CascadeClassifier classifier;
	classifier.load("haarcascades/haarcascade_frontalface_alt_tree.xml");

	ShapePredictor predictor;
	predictor.load("model/helen.txt");

	cout << "face & shape detector loaded\n" << endl;

	FileStorage fs;
	Mat_<float> labels, multihog, ldmks;
	for (int subjId = 1; subjId <= 56; subjId++) {
		if (subjId == testId) continue;
		collectData(subjId, classifier, predictor,
					labels, multihog, ldmks);
	}
	cout << "multihog.rows = " << multihog.rows << endl;
	cout << "multihog.cols = " << multihog.cols << endl;

	// PCA
	cout << "\nbegin PCA" << endl;
	int pcaComp = 400;
	PCA pca(multihog, Mat(), CV_PCA_DATA_AS_ROW, pcaComp);
	fs.open("model/pca.xml", FileStorage::WRITE);
	fs << "mean" << pca.mean;
	fs << "eigenvals" << pca.eigenvalues;
	fs << "eigenvecs" << pca.eigenvectors;
	fs.release();
	cout << "PCA complete" << endl;

	Mat_<float> pcaMat = pca.project(multihog);
	cout << "pcaMat.rows = " << pcaMat.rows << endl;
	cout << "pcaMat.cols = " << pcaMat.cols << endl;

	Mat_<float> dataMat(multihog.rows, pcaMat.cols + ldmks.cols);
	for (int i = 0; i < multihog.rows; i++) {
		for (int j = 0; j < pcaMat.cols; j++)
			dataMat(i, j) = pcaMat(i, j);
		for (int j = 0; j < ldmks.cols; j++)
			dataMat(i, j + pcaMat.cols) = ldmks(i, j);
	}

	// SVM
	cout << "\ntrain SVM" << endl;
	SVMParams params;
	params.svm_type = SVM::C_SVC;
	params.kernel_type = SVM::RBF;

	SVM svm;
	svm.train_auto(dataMat, labels, Mat(), Mat(), params);
	svm.save("model/svm.xml");
	cout << "SVM saved!\n" << endl;
}

void columbiaTest(int testId = 0)
{
	CascadeClassifier classifier;
	classifier.load("haarcascades/haarcascade_frontalface_alt_tree.xml");

	ShapePredictor predictor;
	predictor.load("model/helen.txt");

	PCA pca;
	FileStorage fs("model/pca.xml", FileStorage::READ);
	fs["mean"] >> pca.mean;
	fs["eigenvals"] >> pca.eigenvalues;
	fs["eigenvecs"] >> pca.eigenvectors;
	fs.release();

	/*LDA lda;
	lda.load("model/lda.xml");*/

	SVM svm;
	svm.load("model/svm.xml");

	cout << "\nmodel loaded" << endl;

	// test prediction
	cout << "\nbegin test" << endl;
	int corr = 0, total = 0;

	Mat_<float> labels, multihog, ldmks;
	collectData(testId, classifier, predictor,
				labels, multihog, ldmks);

	for (int i = 0; i < multihog.rows; i++) {
		Mat_<float> pcaVec = pca.project(multihog.row(i));
		Mat_<float> datVec(1, pcaVec.cols + ldmks.cols);
		for (int j = 0; j < pcaVec.cols; j++)
			datVec(0, j) = pcaVec(0, j);
		for (int j = 0; j < ldmks.cols; j++)
			datVec(0, j + pcaVec.cols) = ldmks(i, j);
		//Mat_<float> ldaVec = lda.project(datVec);

		float pred = svm.predict(datVec);
		if ((int)pred == (int)labels(i, 0))
			corr++;

		total++;
	}
	cout << "testId = " << testId << endl;
	cout << "corr = " << corr << " , total = " << total << endl;
	cout << "percentage: " << (double)corr / total << endl;

	ofstream fout("data/testId" + to_string(testId) + ".txt");
	fout << "corr = " << corr << " , total = " << total << endl;
	fout << "percentage: " << (double)corr / total << endl;
	fout.close();
}

void train(set<int> &testSet)
{
	CascadeClassifier classifier;
	classifier.load("haarcascades/haarcascade_frontalface_alt_tree.xml");

	ShapePredictor predictor;
	predictor.load("model/helen.txt");

	cout << "face & shape detector loaded\n" << endl;

	ifstream fin("img/labels.txt");
	string line;
	Mat_<float> labels, multihog, landmarks;
	while (getline(fin, line)) {

		stringstream ss(line);
		int frame, label;
		ss >> frame >> label;
		label -= 49;
		
		if (testSet.find(frame) != testSet.end())
			continue;

		Mat_<uchar> img = imread("img/" + to_string(frame) + ".jpg", 0);
		BBox bbox = getTestBBox(img, classifier);
		if (EmptyBox(bbox)) continue;

		Mat_<float> lbl(1, 1);
		lbl(0, 0) = label;
		labels.push_back(lbl);

		Mat_<double> shape = predictor(img, bbox);
		Geom G;	initGeom(shape, G);
		Pose P; calcPose(G, P);

		Mat_<uchar> lEye, rEye;
		regularize(img, bbox, P, shape, lEye, rEye);

		vector<float> lRlt;
		vector<float> rRlt;
		calcMultiHog(lEye, lRlt);
		calcMultiHog(rEye, rRlt);

		vector<float> _hog2nd_vec;
		for (int k = 0; k < lRlt.size(); k++)
			_hog2nd_vec.push_back(lRlt[k]);
		for (int k = 0; k < rRlt.size(); k++)
			_hog2nd_vec.push_back(rRlt[k]);
		Mat_<float> mhog = Mat_<float>(_hog2nd_vec).reshape(1, 1);
		multihog.push_back(mhog);

		vector<float> _ldmks;
		for (int i = 28; i < 48; i++) {
			_ldmks.push_back((shape(i, 0) - bbox.cx) / bbox.w);
			_ldmks.push_back((shape(i, 1) - bbox.cy) / bbox.h);
		}
		float mouthx = (shape(51, 0) + shape(62, 0) + shape(66, 0) + shape(57, 0)) / 4;
		float mouthy = (shape(51, 1) + shape(62, 1) + shape(66, 1) + shape(57, 1)) / 4;
		_ldmks.push_back((mouthx - bbox.cx) / bbox.w);
		_ldmks.push_back((mouthy - bbox.cy) / bbox.h);
		float maxVal = *std::max_element(_ldmks.begin(), _ldmks.end());
		for (int i = 0; i < _ldmks.size(); i++) _ldmks[i] *= 1.0 / maxVal; // scale to [-1, 1]
		Mat_<float> ldmks = Mat_<float>(_ldmks).reshape(1, 1);
		landmarks.push_back(ldmks);
	}

	// PCA
	cout << "\nbegin PCA" << endl;
	int pcaComp = 400;
	PCA pca(multihog, Mat(), CV_PCA_DATA_AS_ROW, pcaComp);
	FileStorage fs("model/girl_pca.xml", FileStorage::WRITE);
	fs << "mean" << pca.mean;
	fs << "eigenvals" << pca.eigenvalues;
	fs << "eigenvecs" << pca.eigenvectors;
	fs.release();
	cout << "PCA complete" << endl;

	Mat_<float> pcaMat = pca.project(multihog);
	cout << "pcaMat.rows = " << pcaMat.rows << endl;
	cout << "pcaMat.cols = " << pcaMat.cols << endl;

	Mat_<float> dataMat(multihog.rows, pcaMat.cols + landmarks.cols);
	for (int i = 0; i < multihog.rows; i++) {
		for (int j = 0; j < pcaMat.cols; j++)
			dataMat(i, j) = pcaMat(i, j);
		for (int j = 0; j < landmarks.cols; j++)
			dataMat(i, j + pcaMat.cols) = landmarks(i, j);
	}

	// SVM
	cout << "\ntrain SVM" << endl;
	SVMParams params;
	params.svm_type = SVM::C_SVC;
	params.kernel_type = SVM::RBF;

	SVM svm;
	svm.train_auto(dataMat, labels, Mat(), Mat(), params);
	svm.save("model/girl_svm.xml");
	cout << "SVM saved!\n" << endl;
}

void test(set<int> &testSet, int code)
{
	CascadeClassifier classifier;
	classifier.load("haarcascades/haarcascade_frontalface_alt_tree.xml");

	ShapePredictor predictor;
	predictor.load("model/helen.txt");

	PCA pca;
	FileStorage fs("model/girl_pca.xml", FileStorage::READ);
	fs["mean"] >> pca.mean;
	fs["eigenvals"] >> pca.eigenvalues;
	fs["eigenvecs"] >> pca.eigenvectors;

	SVM svm;
	svm.load("model/girl_svm.xml");

	cout << "\nmodel loaded" << endl;

	ifstream fin("img/labels.txt");
	ofstream fout("data/out_" + 
				  to_string(code) + ".txt");
	VideoWriter writer("data/out.avi", 0, 10, Size(1920, 1080), true);

	string line;
	int corr = 0, total = 0;
	while (getline(fin, line)) {
		stringstream ss(line);
		int frame, label;
		ss >> frame >> label;
		label -= 49;

		if (testSet.find(frame) == testSet.end())
			continue;

		Mat vis = imread("img/" + to_string(frame) + ".jpg",
						 CV_LOAD_IMAGE_UNCHANGED);
		Mat_<uchar> img;
		cvtColor(vis, img, COLOR_BGR2GRAY);
		BBox bbox = getTestBBox(img, classifier);
		if (EmptyBox(bbox)) continue;

		Mat_<double> shape = predictor(img, bbox);
		Geom G;	initGeom(shape, G);
		Pose P; calcPose(G, P);

		Mat_<uchar> lEye, rEye;
		regularize(img, bbox, P, shape, lEye, rEye);

		vector<float> lRlt;
		vector<float> rRlt;
		calcMultiHog(lEye, lRlt);
		calcMultiHog(rEye, rRlt);

		Mat_<float> pcaVec, ldmks;

		vector<float> _hog2nd_vec;
		for (int k = 0; k < lRlt.size(); k++)
			_hog2nd_vec.push_back(lRlt[k]);
		for (int k = 0; k < rRlt.size(); k++)
			_hog2nd_vec.push_back(rRlt[k]);
		Mat_<float> multihog = Mat_<float>(_hog2nd_vec).reshape(1, 1);
		pcaVec = pca.project(multihog);

		vector<float> _ldmks;
		for (int i = 28; i < 48; i++) {
			_ldmks.push_back((shape(i, 0) - bbox.cx) / bbox.w);
			_ldmks.push_back((shape(i, 1) - bbox.cy) / bbox.h);
		}
		float mouthx = (shape(51, 0) + shape(62, 0) + shape(66, 0) + shape(57, 0)) / 4;
		float mouthy = (shape(51, 1) + shape(62, 1) + shape(66, 1) + shape(57, 1)) / 4;
		_ldmks.push_back((mouthx - bbox.cx) / bbox.w);
		_ldmks.push_back((mouthy - bbox.cy) / bbox.h);
		float maxVal = *std::max_element(_ldmks.begin(), _ldmks.end());
		for (int i = 0; i < _ldmks.size(); i++) _ldmks[i] *= 1.0 / maxVal; // scale to [-1, 1]
		ldmks = Mat_<float>(_ldmks).reshape(1, 1);

		Mat_<float> sample(1, pcaVec.cols + ldmks.cols);
		for (int j = 0; j < pcaVec.cols; j++)
			sample(0, j) = pcaVec(0, j);
		for (int j = 0; j < ldmks.cols; j++)
			sample(0, j + pcaVec.cols) = ldmks(0, j);

		int pred = svm.predict(sample);
		if (pred == label) corr++;
		total++;

		fout << frame << ' ' << label << ' ' << pred << endl;

		string s1, s2;
		switch (label) {
		case 0: s1 = "annotation: Eye"; break;
		case 1: s1 = "annotation: Face"; break;
		case 2: s1 = "annotation: NOF"; break;
		}
		switch (pred) {
		case 0: s2 = "prediction: Eye"; break;
		case 1: s2 = "prediction: Face"; break;
		case 2: s2 = "prediction: NOF"; break;
		}

		Scalar c1, c2;
		c1 = CV_RGB(255, 255, 0);	// yellow
		if (pred == label) c2 = CV_RGB(0, 255, 0);	// green
		else c2 = CV_RGB(255, 0, 0);				// red

		putText(vis, s1, Point(1280, 100), CV_FONT_HERSHEY_PLAIN, 4.0, c1, 3);
		putText(vis, s2, Point(1280, 200), CV_FONT_HERSHEY_PLAIN, 4.0, c2, 3);
		/*imshow("glance", vis);
		waitKey(0);*/
		writer.write(vis);
	}
	cout << corr << ' ' << total << endl;
	cout << (double)corr / total << endl;
	fin.close();
	fout.close();
	writer.release();
}

int main()
{
	RNG rng(getTickCount());
	set<int> trainSet, demoSet;
	while (trainSet.size() < 3000) {
		int n = rng.uniform(1, 4415);
		trainSet.insert(n);
	}
	for (int i = 1989; i < 3049; i++)
		demoSet.insert(i);
	//train(trainSet);
	test(demoSet, 7);
	/*int imgnum = 4415;
	int k_fold = 5;
	RNG rng(getTickCount());
	for (int i = 0; i < k_fold; i++) {
		set<int> testSet;
		while (testSet.size() < imgnum / k_fold) {
			int num = rng.uniform(1, imgnum);
			testSet.insert(num);
		}
		train(testSet);
		test(testSet, i);
	}*/
	/*ifstream fin("img/labels.txt");
	ofstream fout("img/labels2.txt");
	string line;
	while (getline(fin, line)) {
		stringstream ss(line);
		int frame, label;
		ss >> frame >> label;
		Mat img = imread("img/" + to_string(frame) + ".jpg");
		cout << frame << endl;
		int new_frame = frame + 1988;
		imwrite("img/" + to_string(new_frame) + ".jpg", img);
		fout << new_frame << ' ' << label << endl;
	}
	fin.close();
	fout.close();*/
	return 0;
}