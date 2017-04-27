#include <iostream>
#include<opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include<opencv2\opencv.hpp>

using namespace cv;
using namespace std;

int main() {
	//缩放比例
	int cubeSize = 50;
	int resizeSize = 3 * cubeSize;
	//resizeScale = 0.5;
	//白色的阈值
	double whiteGrayValue = 0.65;
	int whiteSaturationThresh = 50;
	//黑色的阈值
	double blackGrayValue = 0.1;
	//直方图的scale
	int histScale = 100;
	//是否存在跳跃，默认为0，红色不会从1跳到0
	bool HJump = false;
	string FilePath_blue = "../testImage/yellow.jpg";
	string picsPath = "../testImage/";
	//Mat input_image = imread(FilePath, IMREAD_GRAYSCALE);
	vector<string> 	pics = { "blue.jpg", "green.jpg","orange.jpg", "red.jpg" , "white.jpg","yellow.jpg" };
	string picName;
	//存储hist矩阵，并最终合并计算得到同一的阈值
	vector<Mat> hHist, whiteMask, picsHue, picsRGB, grayHist, picsSaturation;

	for (int picNum = 0; picNum < 6; picNum++) {
		picName = pics[picNum];
		Mat input_image, small_image, small_image_gray, small_image_hsv, small_image_bw, small_image_h, small_image_s;
		vector<Mat> channels;
		input_image = imread(picsPath + picName);
		resize(input_image, small_image, Size(resizeSize, resizeSize), 0, 0, 1);

		picsRGB.push_back(small_image);
		//显示图片
		//imshow("RGB "+picName, small_image);
		//应该先得到白色的mask然后再滤波,不然可能会造成logo部位识别不准确
		cvtColor(small_image, small_image_gray, CV_BGR2GRAY);
		GaussianBlur(small_image, small_image, Size(3, 3), 0, 0);
		cvtColor(small_image, small_image_hsv, CV_BGR2HSV_FULL);
		//提取H分量
		split(small_image_hsv, channels);
		small_image_h = channels[0];
		small_image_s = channels[1];
		picsHue.push_back(small_image_h);
		picsSaturation.push_back(small_image_s);
		//得到不含白色的mask
		//threshold(small_image_gray, small_image_bw, whiteGrayValue * 255, 255, CV_THRESH_BINARY_INV);
		small_image_bw = small_image_s >  whiteSaturationThresh;
		whiteMask.push_back(small_image_s < whiteSaturationThresh);
		//H分量的histogram
		int channel = 0;
		Mat dstHist;
		int histSize[] = { 256 };
		float midRanges[] = { 0,255 };
		const float *ranges[] = { midRanges };
		//直方图统计
		calcHist(&small_image_hsv, 1, &channel, small_image_bw, dstHist, 1, histSize, ranges, true, false);
		//calcHist(&small_image_hsv, 1, &channel, Mat(), dstHist, 1, histSize, ranges, true, false);
		hHist.push_back(dstHist);
	}


	Mat hHistTotal;
	hHistTotal = hHist[0] + hHist[1] + hHist[2] + hHist[3] + hHist[4] + hHist[5];
	//显示hHistTotal矩阵
	//cout << "hHistTotal=" << endl << " " << hHistTotal << endl << endl;
	//定义最大噪声，需要根据图片大小确定，图片固定，暂时不用变化
	int	noiseSize = cvRound(0.02*resizeSize*resizeSize);
	//比例大于0.02的mask
	Mat s = hHistTotal > noiseSize;
	//cout << "S=" << endl << s << endl << endl;
	Mat dstHist;// = hHistTotal;
	hHistTotal.copyTo(dstHist, s);

	//	medianBlur(dstHist, dstHist, 15);
	Mat drawImage = Mat::zeros(Size(256, 256), CV_8UC3);
	//像素值最大的归一化
	double HistMaxValue;
	minMaxLoc(dstHist, 0, &HistMaxValue, 0, 0);
	//printf("g_dHistMaxValue=%lf\n", HistMaxValue);
	for (int i = 0; i < 256; i++)
	{
		int value = cvRound(dstHist.at<float>(i) * 256 * 0.9 / HistMaxValue);
		line(drawImage, Point(i, drawImage.rows - 1), Point(i, drawImage.rows - 1 - value), Scalar(255, 255, 255));
	}
	//cout << "dstHist=" << endl << " " << dstHist << endl << endl;
	imshow("hHistTotal", drawImage);

	//定义两种颜色之间的最小gap
	int hGap = 10;
	Mat hIndex = Mat::zeros(Size(1, 256 + 2 * hGap), dstHist.depth());
	for (int i = 0; i < 256 + 2 * hGap; i++)
	{
		if (i < hGap) hIndex.at<float>(i) = dstHist.at<float>(i + 256 - hGap);
		else if (i > 256 - 1 + hGap)hIndex.at<float>(i) = dstHist.at<float>(i - 256 - hGap);
		else hIndex.at<float>(i) = dstHist.at<float>(i - hGap);
	}
	vector<int> hInd;
	int hR = 3;
	for (int i = 0; i < 256; i++) {
		float pixelsSum = 0;
		for (int j = i + hGap - hR; j < i + hGap + hR; j++)
			pixelsSum += hIndex.at<float>(j);
		//printf("hInd[]%f\n", pixelsSum);
		if (pixelsSum > 0.1*resizeSize*resizeSize)
		{
			hInd.push_back(i);
			printf("%d ", i);
		}
	}
	vector<int> hThresh;
	vector<int> hLabel;
	int currentLabel, currentLabelNum, currentLabelSum;
	for (vector<int>::const_iterator citer = hInd.begin(); citer != hInd.end(); citer++)
	{
		//currentLabelSum += *citer;
		if (citer == hInd.begin())
		{
			hLabel.push_back(currentLabel);
			currentLabel = 1;
			currentLabelNum = 1;
			currentLabelSum = 0;

		}
		else if (citer == hInd.end() - 1) {
			hThresh.push_back(cvRound(currentLabelSum / currentLabelNum));
		}
		else
		{
			if ((*(citer)-*(citer - 1)) > hGap)
			{
				hThresh.push_back(cvRound(currentLabelSum / currentLabelNum));
				currentLabelSum = 0;
				currentLabel += 1;
				currentLabelNum = 1;
			}
			else {
				currentLabelNum += 1;
			}
		}
		currentLabelSum += *citer;
	}
	for (vector<int>::const_iterator citer = hThresh.begin(); citer != hThresh.end(); citer++) 
	{
		cout << endl << "hThresh" << *citer << endl;
	}

	//判断红色的hue值是否会发生跳转
	bool isRedHueTwoParts;
	isRedHueTwoParts = ((hThresh.at(hThresh.size() - 1) - hThresh.at(0)) > (256 - hGap));
	cout << endl << isRedHueTwoParts << endl;

	////两端延长了的直方图
	//Mat longDrawImage = Mat::zeros(Size(256 + 2 * hGap, 256), CV_8UC3);
	//minMaxLoc(hIndex, 0, &HistMaxValue, 0, 0);
	//for (int i = 0; i < 256 + 2 * hGap; i++)
	//{
	//	int value = cvRound(hIndex.at<float>(i) * 256 * 0.9 / HistMaxValue);
	//	line(longDrawImage, Point(i, longDrawImage.rows - 1), Point(i, longDrawImage.rows - 1 - value), Scalar(255, 255, 255));
	//}
	//imshow("longHistTotal", longDrawImage);

	//显示图片
	for (int i = 0; i != picsHue.size(); i++) {
		//imshow("Hue "+pics[i], picsHue[i]); 
		//imshow("White " + pics[i], whiteMask[i]);
		Mat picWhitoutWhite;
		picsRGB[i].copyTo(picWhitoutWhite, 255-whiteMask[i]);
		imshow("RGB " + pics[i], picWhitoutWhite);
		//Rect r(0, 0, cubeSize, cubeSize);
		//imshow("RGB " + pics[i], Mat(picsRGB[i], r));
	}

	//建立模型
	



	//system("kcube ");

	waitKey(0);
	//system("pause");

	return 0;
}