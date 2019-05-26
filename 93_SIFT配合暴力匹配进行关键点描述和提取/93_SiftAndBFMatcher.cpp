#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
using namespace cv;
using namespace std;

int main()
{

	//【4】创建视频对象、定义帧率
	//VideoCapture cap(0);
	//VideoCapture cap1(1);
	unsigned int frameCount = 0;//帧数

	//【5】不断循环，直到q键被按下
	/*while(1)
	{*/
		//【1】载入图像、显示并转化为灰度图
		Mat trainImage;
		//cap1 >> trainImage;
		trainImage = imread("E:sift\\salt.jpg");
		resize(trainImage, trainImage, Size(400, 400));
		Mat trainImage_gray;
		//imshow("原始图",trainImage);
		//resize(trainImage, trainImage, Size(400, 400), 0, 0, 1);

		cvtColor(trainImage, trainImage_gray, CV_BGR2GRAY);

		//【2】检测SIFT关键点、提取训练图像描述符
		vector<KeyPoint> train_keyPoint;
		Mat trainDescription;
		SiftFeatureDetector featureDetector;
		featureDetector.detect(trainImage_gray, train_keyPoint);
		SiftDescriptorExtractor featureExtractor;
		featureExtractor.compute(trainImage_gray, train_keyPoint, trainDescription);

		// 【3】进行基于描述符的暴力匹配
		BFMatcher matcher;
		vector<Mat> train_desc_collection(1, trainDescription);
		matcher.add(train_desc_collection);
		matcher.train();



		//<1>参数设置

		Mat captureImage, captureImage_gray;
		//cap >> captureImage;//采集视频到testImage中
		captureImage = imread("E:\\sift\\15.jpg");
		resize(captureImage, captureImage, Size(400, 400));
		/*if(captureImage.empty())
			continue;*/

		//<2>转化图像到灰度
		cvtColor(captureImage, captureImage_gray, CV_BGR2GRAY);

		//<3>检测SURF关键点、提取测试图像描述符
		vector<KeyPoint> test_keyPoint;
		Mat testDescriptor;
		featureDetector.detect(captureImage_gray, test_keyPoint);
		featureExtractor.compute(captureImage_gray, test_keyPoint, testDescriptor);

		//<4>匹配训练和测试描述符
		vector<vector<DMatch> > matches;
		matcher.knnMatch(testDescriptor, matches, 2);

		// <5>根据劳氏算法（Lowe's algorithm），得到优秀的匹配点
		vector<DMatch> goodMatches;
		for(unsigned int i = 0; i < matches.size(); i++)
		{
			if(matches[i][0].distance < 1 * matches[i][1].distance)
				goodMatches.push_back(matches[i][0]);
		}

		//<6>绘制匹配点并显示窗口
		Mat dstImage;
		drawMatches(captureImage, test_keyPoint, trainImage, train_keyPoint, goodMatches, dstImage, Scalar(0,0,255));
		imshow("匹配窗口", dstImage);
		static int i = 0;
		cout << i << endl;
		i++;
		waitKey(0);
		
		
	//}

	return 0;
}
