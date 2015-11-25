#include "GazeEstimation.h"
#include "FaceAlignment.h"
#include "PoseEstimation.h"
using namespace std;
using namespace cv;

Mat_<double> readShape(string filename, int ldmkNum)
{
	FILE *file = fopen(filename.c_str(), "r");
	if (!file) {
		fprintf(stderr, "Unable to read file");
		exit(-1);
	}

	// ignore the first 5 items;
	char tmp[128];
	for (int i = 0; i < 5; i++) {
		fscanf(file, "%s", tmp);
	}

	// read landmarks
	Mat_<double> shape = Mat_<double>::zeros(ldmkNum, 2);
	for (int i = 0; i < ldmkNum; i++)
		fscanf(file, "%lf%lf", &shape(i, 0), &shape(i, 1));

	fclose(file);
	return shape;
}

BBox getTrainBBox(Mat_<uchar> &gray, Mat_<double> &shape, CascadeClassifier &classifier)
{
	double minx, miny, maxx, maxy;
	minMaxLoc(shape.col(0), &minx, &maxx);
	minMaxLoc(shape.col(1), &miny, &maxy);
	double w = maxx - minx;
	double h = maxy - miny;
	double scale = 2;
	minx = std::max(0.0, minx - (scale - 1)*w / 2);
	miny = std::max(0.0, miny - (scale - 1)*h / 2);
	maxx = std::min(gray.cols - 1.0, minx + scale*w);
	maxy = std::min(gray.rows - 1.0, miny + scale*h);
	w = maxx - minx;
	h = maxy - miny;

	Mat_<uchar> faceImg = gray(Rect(minx, miny, w, h));
	vector<Rect> faces;
	equalizeHist(faceImg, faceImg);
	classifier.detectMultiScale(faceImg, faces, 1.05, 2,
								CV_HAAR_FIND_BIGGEST_OBJECT,
								Size(w / 4, h / 4));
	/*if (faces.empty()) {
		CascadeClassifier prof;
		prof.load("haarcascades/haarcascade_profileface.xml");
		prof.detectMultiScale(faceImg, faces, 1.2, 3,
							  CV_HAAR_FIND_BIGGEST_OBJECT |
							  CV_HAAR_DO_CANNY_PRUNING,
							  Size(w / 4, h / 4));
	}*/

	BBox bbox;
	if (faces.empty()) {
		bbox.x = 0;
		bbox.y = 0;
		bbox.w = 0;
		bbox.h = 0;
	}
	else {
		bbox.x = minx + faces[0].x;
		bbox.y = miny + faces[0].y;
		bbox.w = faces[0].width;
		bbox.h = faces[0].height;
	}
	bbox.cx = bbox.x + bbox.w / 2.0;
	bbox.cy = bbox.y + bbox.h / 2.0;
	return bbox;
}

BBox getTestBBox(Mat_<uchar> &gray, CascadeClassifier &classifier)
{
	Mat_<uchar> mgray;
	double scale = 300.0 / gray.rows;
	resize(gray, mgray, Size(), scale, scale);

	Mat_<uchar> egray;
	vector<Rect> faces;
	equalizeHist(mgray, egray);
	classifier.detectMultiScale(egray, faces, 1.05, 2,
								CV_HAAR_FIND_BIGGEST_OBJECT
								| CV_HAAR_DO_CANNY_PRUNING);
	/*if (faces.empty()) {
		CascadeClassifier prof;
		prof.load("haarcascades/haarcascade_profileface.xml");
		prof.detectMultiScale(egray, faces, 1.2, 3,
							  CV_HAAR_FIND_BIGGEST_OBJECT |
							  CV_HAAR_DO_CANNY_PRUNING);
	}*/

	BBox bbox;
	if (faces.empty()) {
		bbox.x = 0;
		bbox.y = 0;
		bbox.w = 0;
		bbox.h = 0;
	}
	else {
		bbox.x = faces[0].x / scale;
		bbox.y = faces[0].y / scale;
		bbox.w = faces[0].width / scale;
		bbox.h = faces[0].height / scale;
		bbox.cx = bbox.x + bbox.w / 2.0;
		bbox.cy = bbox.y + bbox.h / 2.0;
	}
	return bbox;
}

Rect findEyeRect(Mat_<uchar> &gray, Mat_<double> &shape, int whichEye)
{
	int idx1, idx2;
	if (whichEye == LEFT_EYE) {
		idx1 = 36;
		idx2 = 42;
	}
	else if (whichEye == RIGHT_EYE) {
		idx1 = 42;
		idx2 = 48;
	}

	Mat_<double> tmp(idx2 - idx1, 2);
	for (int i = idx1; i < idx2; i++) {
		tmp(i - idx1, 0) = shape(i, 0);
		tmp(i - idx1, 1) = shape(i, 1);
	}

	double minx, miny, maxx, maxy;
	minMaxLoc(tmp.col(0), &minx, &maxx);
	minMaxLoc(tmp.col(1), &miny, &maxy);
	double w = maxx - minx;
	double h = maxy - miny;
	double cx = minx + 0.5*w;
	double cy = miny + 0.5*h;

	if (h < 0.5*w) {
		w *= 1.1;
		h = 0.5*w;
	}
	else {
		h *= 1.1;
		w = 2 * h;
	}
	minx = cx - 0.5*w;
	miny = cy - 0.5*h;

	return Rect(minx, miny, w, h);
}

vector<Point> findEyeCont(Rect &EyeRect, Mat_<double> &shape, int whichEye)
{
	Mat_<double> EyeShape(6, 2);

	int idx1, idx2;
	if (whichEye == LEFT_EYE) {
		idx1 = 36;
		idx2 = 42;
	}
	else if (whichEye == RIGHT_EYE) {
		idx1 = 42;
		idx2 = 48;
	}
	for (int i = idx1; i < idx2; i++) {
		EyeShape(i - idx1, 0) = shape(i, 0) - EyeRect.x;
		EyeShape(i - idx1, 1) = shape(i, 1) - EyeRect.y;
	}

	Mat_<float> srcX(4, 1);
	Mat_<float> srcY(4, 1);
	Mat_<float> uP(4, 1);	// upper poly
	Mat_<float> lP(4, 1);	// lower poly

							// fit upper eyelid
	for (int i = 0; i < 4; i++) {
		srcX(i, 0) = EyeShape(i, 0);
		srcY(i, 0) = EyeShape(i, 1);
	}
	polyfit(srcX, srcY, uP, 3);
	// fit lower eyelid
	for (int i = 3; i < 6; i++) {
		srcX(i - 3, 0) = EyeShape(i, 0);
		srcY(i - 3, 0) = EyeShape(i, 1);
	}
	srcX(3, 0) = EyeShape(0, 0);
	srcY(3, 0) = EyeShape(0, 1);
	polyfit(srcX, srcY, lP, 3);

	// create eyelids contour
	vector<Point> contour;
	double xstep = (EyeShape(3, 0) - EyeShape(0, 0)) / 10;
	// upper contour
	contour.push_back(Point(EyeShape(0, 0), EyeShape(0, 1)));
	for (int i = 1; i < 10; i++) {
		double x = EyeShape(0, 0) + i*xstep;
		double y = uP(3, 0)*x*x*x + uP(2, 0)*x*x + uP(1, 0)*x + uP(0, 0);
		contour.push_back(Point(x, y));
	}
	// lower contour
	contour.push_back(Point(EyeShape(3, 0), EyeShape(3, 1)));
	for (int i = 9; i > 0; i--) {
		double x = EyeShape(0, 0) + i*xstep;
		double y = lP(3, 0)*x*x*x + lP(2, 0)*x*x + lP(1, 0)*x + lP(0, 0);
		contour.push_back(Point(x, y));
	}

	return contour;
}

Mat_<uchar> findEyeMask(Mat_<uchar> &EyeGray, vector<Point> &contour)
{
	// create mask
	Mat_<uchar> mask = Mat_<uchar>::zeros(EyeGray.size());
	vector<vector<Point> > contAll;
	contAll.push_back(contour);
	fillPoly(mask, contAll, Scalar(255), 8);

	return mask;
}

void helenTrain()
{
	int imgnum = 2000;
	int ldmkNum = 68;

	vector<Mat_<uchar> > images;
	vector<Mat_<double> > shapes;
	vector<BBox> bboxes;
	images.reserve(imgnum);
	shapes.reserve(imgnum);
	bboxes.reserve(imgnum);

	CascadeClassifier classifier;
	classifier.load("haarcascades/haarcascade_frontalface_alt_tree.xml");

	for (int i = 0; i < imgnum; i++) {
		string imgname = "helen/trainset/" + to_string(i) + ".jpg";
		string ptsname = "helen/trainset/" + to_string(i) + ".pts";

		Mat_<uchar> img = imread(imgname, 0);
		if (img.empty()) {
			cout << imgname << " doesn't exist!" << endl;
			continue;
		}
		Mat_<double> shape = readShape(ptsname, ldmkNum);
		BBox bbox = getTrainBBox(img, shape, classifier);
		if (EmptyBox(bbox)) {
			cout << "no face detected in " << imgname << endl;
			continue;
		}

		images.push_back(img);
		shapes.push_back(shape);
		bboxes.push_back(bbox);
	}
	images.shrink_to_fit();
	shapes.shrink_to_fit();
	bboxes.shrink_to_fit();
	printf("\nActual Training Set Size: %d\n", (int)images.size());

	ShapeTrainer trainer;
	ShapePredictor predictor = trainer.train(images, shapes, bboxes);
	cout << "Saving model..." << endl;
	predictor.save("model/helen.txt");

	cout << "Complete!" << endl;
}

void helenTest()
{
	int imgnum = 330;
	int ldmkNum = 68;

	CascadeClassifier classifier;
	classifier.load("haarcascades/haarcascade_frontalface_alt_tree.xml");

	ShapePredictor predictor;
	predictor.load("model/helen.txt");

	for (int i = 0; i < imgnum; i++) {
		string imgname = "helen/testset/" + to_string(i) + ".jpg";

		Mat_<uchar> img = imread(imgname, 0);
		if (img.empty()) {
			cout << imgname << " doesn't exist!" << endl;
			continue;
		}
		BBox bbox = getTestBBox(img, classifier);
		if (EmptyBox(bbox)) {
			cout << "no face detected in " << imgname << endl;
			continue;
		}

		Mat_<double> shape = predictor(img, bbox);

		Geom G;
		Pose P;
		initGeom(shape, G);
		calcPose(G, P);

		Mat regface;
		Mat_<double> newshape;
		//regularize(img, shape, bbox, P.roll, regface, newshape);

		int maxedge = std::max(regface.rows, regface.cols);
		if (maxedge > 300) {
			double scale = 300.0 / maxedge;
			resize(regface, regface, Size(), scale, scale);
		}
		imshow("regular face", regface);

		Mat vis = imread(imgname);
		drawShape(vis, shape, 2);
		maxedge = std::max(vis.rows, vis.cols);
		if (maxedge > 600) {
			double scale = 600.0 / maxedge;
			resize(vis, vis, Size(), scale, scale);
		}
		imshow("Prediction", vis);
		char c = waitKey(0);
		if (c == 27) break;
	}
}

void readAllData(ifstream &fin, Mat_<double> &labels, Mat_<double> &data)
{
	string line;
	while (getline(fin, line)) {
		stringstream ss(line);

		int dim;
		Mat_<double> label(1, 1);
		ss >> label(0, 0) >> dim;

		Mat_<double> datarow(1, dim);
		for (int i = 0; i < dim; i++)
			ss >> datarow(0, i);

		labels.push_back(label);
		data.push_back(datarow);
	}
}

void writeDataRow(ofstream &fout,
				  int label,
				  BBox &bbox,
				  Mat_<double> &shape,
				  Mat_<double> &edgePts)
{
	fout << label << ' ';
	fout << (21 + edgePts.rows) * 2 << ' ';

	for (int i = 28; i < 48; i++) {
		fout << shape(i, 0) - bbox.cx << ' '
			<< shape(i, 1) - bbox.cy << ' ';
	}

	double mouthx = (shape(51, 0) + shape(62, 0) + shape(66, 0) + shape(57, 0)) / 4;
	double mouthy = (shape(51, 1) + shape(62, 1) + shape(66, 1) + shape(57, 1)) / 4;
	fout << mouthx - bbox.cx << ' ' << mouthy - bbox.cy << ' ';

	for (int i = 0; i < edgePts.rows; i++) {
		fout << edgePts(i, 0) - bbox.cx << ' '
			<< edgePts(i, 1) - bbox.cy << ' ';
	}

	fout << endl;
}

/*
数据标定说明：
标签为数字1-9，每个数字代表一个注视方向（类似数字小键盘）
	7(左上)  8(上)  9(右上)
	4(左)	5(对视)   6(右)
	1(左下)  2(下)  3(右下)
对于不好区分方向或者数据明显有误的帧，可以按0跳过
按ESC提前结束标定

*这里的对视定义为看着鼻子以上的区域即可
*/
void truthCalib()
{
	cout << "Loading data" << endl;

	CascadeClassifier classifier;
	classifier.load("haarcascades/haarcascade_frontalface_alt_tree.xml");

	ShapePredictor predictor;
	predictor.load("model/helen.txt");

	VideoCapture capture;
	capture.open("data/0013.m4v");

	ofstream fout;
	fout.open("data/0013_all.txt");

	cout << "Complete!\n" << endl;

	while (true) {
		Mat frame;
		capture >> frame;
		if (frame.empty()) break;

		Mat_<uchar> gray;
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		BBox bbox = getTestBBox(gray, classifier);
		if (EmptyBox(bbox)) continue;

		// shape regression
		Mat_<double> shape = predictor(gray, bbox);

		// pose estimation
		Geom G;
		Pose P;
		initGeom(shape, G);
		calcPose(G, P);

		// primary edge detection
		Mat_<double> edgePts;
		/*if (!primaryEdge(gray, bbox, shape, P, edgePts, 30)) {
			cout << "No primary edge detected" << endl;
			continue;
		}*/

		for (int i = 0; i < edgePts.rows; i++) {
			Point pt(edgePts(i, 0), edgePts(i, 1));
			circle(frame, pt, 1, CV_RGB(255, 0, 0));
		}
		drawShape(frame, shape, 1);

		int maxedge = std::max(frame.rows, frame.cols);
		if (maxedge > 1200) {
			double scale = 1200.0 / maxedge;
			resize(frame, frame, Size(), scale, scale);
		}
		imshow("truth calib", frame);

		char c = waitKey(0);
		if (c == 27) break;
		else if (c == '0') continue;
		else if ('1' <= c && c <= '9')
			writeDataRow(fout, c - '0', bbox, shape, edgePts);
		else cout << "Please input correct label" << endl;
	}
}