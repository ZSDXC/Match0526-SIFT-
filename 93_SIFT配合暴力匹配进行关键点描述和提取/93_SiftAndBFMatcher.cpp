#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
using namespace cv;
using namespace std;

int main()
{

	//��4��������Ƶ���󡢶���֡��
	//VideoCapture cap(0);
	//VideoCapture cap1(1);
	unsigned int frameCount = 0;//֡��

	//��5������ѭ����ֱ��q��������
	/*while(1)
	{*/
		//��1������ͼ����ʾ��ת��Ϊ�Ҷ�ͼ
		Mat trainImage;
		//cap1 >> trainImage;
		trainImage = imread("E:sift\\salt.jpg");
		resize(trainImage, trainImage, Size(400, 400));
		Mat trainImage_gray;
		//imshow("ԭʼͼ",trainImage);
		//resize(trainImage, trainImage, Size(400, 400), 0, 0, 1);

		cvtColor(trainImage, trainImage_gray, CV_BGR2GRAY);

		//��2�����SIFT�ؼ��㡢��ȡѵ��ͼ��������
		vector<KeyPoint> train_keyPoint;
		Mat trainDescription;
		SiftFeatureDetector featureDetector;
		featureDetector.detect(trainImage_gray, train_keyPoint);
		SiftDescriptorExtractor featureExtractor;
		featureExtractor.compute(trainImage_gray, train_keyPoint, trainDescription);

		// ��3�����л����������ı���ƥ��
		BFMatcher matcher;
		vector<Mat> train_desc_collection(1, trainDescription);
		matcher.add(train_desc_collection);
		matcher.train();



		//<1>��������

		Mat captureImage, captureImage_gray;
		//cap >> captureImage;//�ɼ���Ƶ��testImage��
		captureImage = imread("E:\\sift\\15.jpg");
		resize(captureImage, captureImage, Size(400, 400));
		/*if(captureImage.empty())
			continue;*/

		//<2>ת��ͼ�񵽻Ҷ�
		cvtColor(captureImage, captureImage_gray, CV_BGR2GRAY);

		//<3>���SURF�ؼ��㡢��ȡ����ͼ��������
		vector<KeyPoint> test_keyPoint;
		Mat testDescriptor;
		featureDetector.detect(captureImage_gray, test_keyPoint);
		featureExtractor.compute(captureImage_gray, test_keyPoint, testDescriptor);

		//<4>ƥ��ѵ���Ͳ���������
		vector<vector<DMatch> > matches;
		matcher.knnMatch(testDescriptor, matches, 2);

		// <5>���������㷨��Lowe's algorithm�����õ������ƥ���
		vector<DMatch> goodMatches;
		for(unsigned int i = 0; i < matches.size(); i++)
		{
			if(matches[i][0].distance < 1 * matches[i][1].distance)
				goodMatches.push_back(matches[i][0]);
		}

		//<6>����ƥ��㲢��ʾ����
		Mat dstImage;
		drawMatches(captureImage, test_keyPoint, trainImage, train_keyPoint, goodMatches, dstImage, Scalar(0,0,255));
		imshow("ƥ�䴰��", dstImage);
		static int i = 0;
		cout << i << endl;
		i++;
		waitKey(0);
		
		
	//}

	return 0;
}
