#include <stdio.h>

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <dirent.h>
#include <thread>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"
using namespace std;
using namespace cv;

#define IMG_PATH_L              "/home/cbreezy/baidunetdiskdownload/stereo_img_zhijiang/No1zhijaing/image_0/"
#define IMG_PATH_R              "/home/cbreezy/baidunetdiskdownload/stereo_img_zhijiang/No1zhijaing/image_1/"
#define IMG_NAME_FILE_PATH      "../config_data/zhijiang_file_name.txt"
#define IMG_NAME_SAVE_PATH      "../config_data/zhijiang_save_name.txt"
#define SAVE_DISPARITY_PATH     "/home/cbreezy/baidunetdiskdownload/stereo_img_zhijiang/No1zhijaing/disparity/"
#define SAVE_DEPTH_PATH         "/home/cbreezy/baidunetdiskdownload/stereo_img_zhijiang/No1zhijaing/depth/"
#define SAVE_RECTIFY_PATH_L     "/home/cbreezy/baidunetdiskdownload/stereo_img_zhijiang/No1zhijaing/rectify_0/"
#define SAVE_RECTIFY_PATH_R     "/home/cbreezy/baidunetdiskdownload/stereo_img_zhijiang/No1zhijaing/rectify_1/"

/* image save flag */
#define SAVE_DIS_IMAGES           0
#define SAVE_DEPTH_IMAGES         0

#define DEPTH_SCALE_FACTOR        32
#define DISPARITY_SCALE_FACTOR    16
float N_WAITEKEY = 0;

/* fx & baseline */
float fx = 363.5;
float baseline = 20; 

const int imageWidth = 640;                             //摄像头的分辨率
const int imageHeight = 480;
const Size imageSize = Size(imageWidth, imageHeight);

Mat R21, t21;                                           //R_21, t_21: R21 * pose -> from cam1 to cam2
Mat Rl, Rr, Pl, Pr, Q;                                  //校正旋转矩阵R，投影矩阵P 重投影矩阵Q
Mat mapLx, mapLy, mapRx, mapRy;                         //映射表
Rect validROIL, validROIR;                              //裁剪之后的区域
Mat xyz;
Mat rectifyImageL, rectifyImageR;                       //存放对齐后的图像
Mat depthMap_16UC1, depthMap;

Mat cameraMatrixL = (Mat_<float>(3, 3) <<
                        363.521634, 0, 320.812941, 
                        0, 363.964555, 238.310163, 
                        0, 0, 1);

Mat distCoeffL = (Mat_<float>(4, 1) <<0.066295, -0.048028,
                      -0.002303, -0.000420);

Mat cameraMatrixR = (Mat_<float>(3, 3) <<
                        360.833487, 0, 334.388312, 
                        0, 361.311662, 242.571617, 
                        0, 0, 1);

Mat distCoeffR = (Mat_<float>(4, 1) << 0.069053, -0.054059,
                      -0.001346, -0.001661);

Mat cam1_T_cam0 = (Mat_<double>(4, 4) <<
                        0.99995005, 0.0087959152, -0.0047494778, -0.19627762,
                        -0.0087504247, 0.99991643, 0.0095153786, -0.00092906068,
                        0.0048327777, -0.0094733434, 0.99994344, -0.0042496733,
                        0, 0, 0, 1);

void insertDepth32f(Mat& depth);
void Loadstereoimg(const string &TxtFilename, vector<string> &vstrImageFilenames);
void image_process(int num_begin, int num_end, vector<string> vstrImageFilenames, vector<string> vstrImage_SaveNames);
void gray2color(Mat img_gray, Mat img_color);
void onMouse(int event, int x, int y,int,void*);

int main(int argc, char **argv)
{
    R21 = (Mat_<double>(3, 3) <<
                            cam1_T_cam0.at<double>(0,0),
                            cam1_T_cam0.at<double>(0,1),
                            cam1_T_cam0.at<double>(0,2),
                            cam1_T_cam0.at<double>(1,0),
                            cam1_T_cam0.at<double>(1,1),
                            cam1_T_cam0.at<double>(1,2),
                            cam1_T_cam0.at<double>(2,0),
                            cam1_T_cam0.at<double>(2,1),
                            cam1_T_cam0.at<double>(2,2));
    t21 = (Mat_<double>(3, 1) <<
                            cam1_T_cam0.at<double>(0,3),
                            cam1_T_cam0.at<double>(1,3),
                            cam1_T_cam0.at<double>(2,3));

    string TxtFilename = string(IMG_NAME_FILE_PATH);
	vector<string> vstrImageFilenames, vstrImage_SaveNames;
    Loadstereoimg(TxtFilename, vstrImageFilenames);
    Loadstereoimg(IMG_NAME_SAVE_PATH, vstrImage_SaveNames);
    int number_img = vstrImageFilenames.size();
    cout << "number of images is: " << number_img <<endl;
    //string name = vstrImageFilenames.back();  // check the back of the image_name
    //cout << name << endl;
    //Size newSize(static_cast<int>(imageWidth*1.2), static_cast<int>(imageHeight*1.2));
    Size newSize(imageWidth, imageHeight);
    stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R21, t21, Rl, Rr, Pl, Pr, Q,
        CALIB_ZERO_DISPARITY, 1, imageSize, &validROIL, &validROIR);
    initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, newSize,
                                    CV_32FC1, mapLx, mapLy);
    initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, newSize,
                                    CV_32FC1, mapRx, mapRy);

    cout << "cameraMatrixL" << cameraMatrixL << endl;
    cout << "cameraMatrixR" << cameraMatrixR << endl;
    cout << "Pl" << Pl << endl;
    cout << "Pr" << Pr << endl;
    cout << "Q" << Rl << endl;
/*
    int process_begin = atoi(argv[1]);
    int process_end   = atoi(argv[2]);
    int num_first,num_second;
    num_first = process_begin + (process_end - process_begin)/3;
    num_second = num_first + (process_end - process_begin)/3;


    thread task10(image_process, process_begin,num_first,vstrImageFilenames,compression_params);
    thread task20(image_process, num_first,num_second,vstrImageFilenames,compression_params);
    thread task30(image_process, num_second,process_end,vstrImageFilenames,compression_params);
    task10.join();
    task20.join();
    task30.join();
*/
    image_process(618,619,vstrImageFilenames,vstrImage_SaveNames);
    cout << "-------------------------------------------"<<endl<<
    "image_process is done..." << endl;
    return 0;
}

void image_process(int num_begin, int num_end, vector<string> vstrImageFilenames, vector<string> vstrImage_SaveNames)
{
    for(int i=num_begin;i<num_end;i++)
    {
        Mat imageL, imageR;
        cout << "it's working on the " << i << " image." << endl;

        imageL = imread(IMG_PATH_L + vstrImageFilenames[i], 0); // 视差处理的图像必须为gray（0）; 不输入默认为1,color
        imageR = imread(IMG_PATH_R + vstrImageFilenames[i], 0);
        imshow("imageL",imageL);
        waitKey(N_WAITEKEY);

        remap(imageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR);
        remap(imageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR);

// 输出矫正好的图像
        //string save_rectify_l = SAVE_RECTIFY_PATH_L + vstrImageFilenames[i];
        //string save_rectify_r = SAVE_RECTIFY_PATH_R + vstrImageFilenames[i];
        //imwrite(save_rectify_l, rectifyImageL);
        //imwrite(save_rectify_r, rectifyImageR);
// 输出矫正好的图像

        //显示矫正后的图像

        for(int i=20; i<rectifyImageL.rows; i+=20)
        {
            line(rectifyImageL,Point(0,i),Point(rectifyImageL.cols,i),Scalar(255,255,255));
            line(rectifyImageR,Point(0,i),Point(rectifyImageL.cols,i),Scalar(255,255,255));
        }
        Mat imageMatches;
        drawMatches(rectifyImageL, vector<KeyPoint>(),  // 1st image
            rectifyImageR, vector<KeyPoint>(),              // 2nd image
            vector<DMatch>(),
            imageMatches,                       // the image produced
            Scalar(255, 255, 255),
            Scalar(255, 255, 255),
            vector<char>(),
            2);
        imshow("imageMatches", imageMatches);
        //waitKey(N_WAITEKEY);

        Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,16,3);
        int sgbmWinSize = 11; //11
        int cn = imageL.channels();
        //int numberOfDisparities = ((imageSize.width/8) + 15) & -16;
        //int numberOfDisparities = 128;//128

        sgbm->setPreFilterCap(63);
        sgbm->setBlockSize(sgbmWinSize);
        sgbm->setP1(8*cn*sgbmWinSize*sgbmWinSize);
        sgbm->setP2(32*cn*sgbmWinSize*sgbmWinSize);
        sgbm->setMinDisparity(0);
        sgbm->setNumDisparities(32);// numberOfDisparities
        sgbm->setUniquenessRatio(10);
        sgbm->setSpeckleWindowSize(200); // 200
        sgbm->setSpeckleRange(32);
        sgbm->setDisp12MaxDiff(1);
        sgbm->setMode(StereoSGBM::MODE_SGBM);

        Mat disparity_image, disparity_image_8UC1, disparity_image_32F;
        sgbm->compute(rectifyImageL, rectifyImageR, disparity_image);
        disparity_image.convertTo(disparity_image_32F, CV_32F, 1.0/16);

        // fill holes
        insertDepth32f(disparity_image_32F);
        //disparity_image_32F.convertTo(disparity_image_8UC1, CV_8U, 255/(numberOfDisparities*8.)*16*2);
        disparity_image_32F.convertTo(disparity_image_8UC1, CV_8U, DISPARITY_SCALE_FACTOR);

        Mat img_pseudocolor(disparity_image_8UC1.rows, disparity_image_8UC1.cols, CV_8UC3);//构造RGB图像，参数CV_8UC3教程文档里面有讲解
 
         int tmp=0;
         for (int y=0;y<disparity_image_8UC1.rows;y++)//转为伪彩色图像的具体算法
         {
                for (int x=0;x<disparity_image_8UC1.cols;x++)
                {
                    tmp = disparity_image_8UC1.at<unsigned char>(y,x);
                    img_pseudocolor.at<Vec3b>(y,x)[0] = abs(255-tmp); //blue
                    img_pseudocolor.at<Vec3b>(y,x)[1] = abs(127-tmp); //green
                    img_pseudocolor.at<Vec3b>(y,x)[2] = abs( 0-tmp); //red
                }
         }

        imshow("disparity_image", img_pseudocolor);
        //cv::reprojectImageTo3D(disparity_image_32F, xyz, Q, true);
        //xyz = xyz * 16; // xyz=[X/W Y/W Z/W]，乘以16得到真实坐标
        //cv::setMouseCallback("disparity_image", onMouse, 0);
        //waitKey();
        // no fill holes
        //disparity_image.convertTo(disparity_image_8UC1, CV_8U, 255/(numberOfDisparities*8.)*4);
        //imshow("disparity_image", disparity_image_8UC1);
        //waitKey(N_WAITEKEY);

        if (SAVE_DIS_IMAGES) 
        {
            string save_disparity_url = SAVE_DISPARITY_PATH + vstrImageFilenames[i];
            imwrite(save_disparity_url, disparity_image_8UC1);
        }
        
        /// comtute the depth
        depthMap = Mat::zeros(disparity_image_32F.size(), CV_32FC1);
        int height = disparity_image_32F.rows;
        int width = disparity_image_32F.cols;
        for(int k = 0;k < height; k++)
        {
            //const float* inData = disparity_image_32F.ptr<float>(k);
            float* inData = disparity_image_32F.ptr<float>(k);
            float* outData = depthMap.ptr<float>(k);
            for(int i = 0; i < width; i++)
            {
                //if(!inData[i]||inData[i]<1) 
                if(!inData[i]) 
                    inData[i] = 1;
                outData[i] = float(fx *baseline / inData[i]);
            }
        }
        //Mat depthMap_8UC1 = Mat(depthMap.rows, depthMap.cols, CV_8UC1);
        //normalize(depthMap, depthMap_8UC1, 0, 255, NORM_MINMAX, CV_8U);
        medianBlur(depthMap, depthMap, 7);
        Mat depthMap_8UC1;
        depthMap.convertTo(depthMap_8UC1, CV_8UC1, 1.0/DEPTH_SCALE_FACTOR);
        depthMap.convertTo(depthMap_16UC1, CV_16UC1);
        // depth_img filter

        Mat img = depthMap_8UC1.clone();
        Mat img_color(img.rows, img.cols, CV_8UC3);//构造RGB图像
        gray2color(img, img_color);

        imshow("depthMap", img_color);
        cv::setMouseCallback("depthMap", onMouse, 0);
        waitKey(N_WAITEKEY);


        if (SAVE_DEPTH_IMAGES) 
        {
            //depthMap_8UC1 = depthMap_8UC1/32;
            //string save_depth_url = SAVE_DEPTH_PATH + vstrImageFilenames[i];
            string save_depth_url = SAVE_DEPTH_PATH + vstrImage_SaveNames[i];
            imwrite(save_depth_url, depthMap_16UC1);
        }

    /*
        for(int x = 50; x< 450;x+=25)
            for(int y =50;y<650;y+=25)
                cout << depthMap_16UC1.ptr<ushort>(x)[y] << endl;
    */


    }
}

void Loadstereoimg(const string &TxtFilename, vector<string> &vstrImageFilenames)
{
    ifstream f;
    f.open(TxtFilename.c_str());
    while(!f.eof())
    {
        string s;
        getline(f,s);
        if(!s.empty())
            vstrImageFilenames.push_back(s);
    }
}

void insertDepth32f(Mat& depth)
{
    const int width = depth.cols;
    const int height = depth.rows;
    float* data = (float*)depth.data;
    Mat integralMap = Mat::zeros(height, width, CV_64F);
    Mat ptsMap = Mat::zeros(height, width, CV_32S);
    double* integral = (double*)integralMap.data;
    int* ptsIntegral = (int*)ptsMap.data;
    memset(integral, 0, sizeof(double) * width * height);
    memset(ptsIntegral, 0, sizeof(int) * width * height);
    for (int i = 0; i < height; ++i)
    {
        int id1 = i * width;
        for (int j = 0; j < width; ++j)
        {
            int id2 = id1 + j;
            if (data[id2] > 1e-3)
            {
                integral[id2] = data[id2];
                ptsIntegral[id2] = 1;
            }
        }
    }

    for (int i = 0; i < height; ++i)
    {
        int id1 = i * width;
        for (int j = 1; j < width; ++j)
        {
            int id2 = id1 + j;
            integral[id2] += integral[id2 - 1];
            ptsIntegral[id2] += ptsIntegral[id2 - 1];
        }
    }
    for (int i = 1; i < height; ++i)
    {
        int id1 = i * width;
        for (int j = 0; j < width; ++j)
        {
            int id2 = id1 + j;
            integral[id2] += integral[id2 - width];
            ptsIntegral[id2] += ptsIntegral[id2 - width];
        }
    }
    int wnd;
    double dWnd = 2;
    while (dWnd > 1)
    {
        wnd = int(dWnd);
        dWnd /= 2;
        for (int i = 0; i < height; ++i)
        {
            int id1 = i * width;
            for (int j = 0; j < width; ++j)
            {
                int id2 = id1 + j;
                int left = j - wnd - 1;
                int right = j + wnd;
                int top = i - wnd - 1;
                int bot = i + wnd;
                left = max(0, left);
                right = min(right, width - 1);
                top = max(0, top);
                bot = min(bot, height - 1);
                int dx = right - left;
                int dy = (bot - top) * width;
                int idLeftTop = top * width + left;
                int idRightTop = idLeftTop + dx;
                int idLeftBot = idLeftTop + dy;
                int idRightBot = idLeftBot + dx;
                int ptsCnt = ptsIntegral[idRightBot] + ptsIntegral[idLeftTop] - (ptsIntegral[idLeftBot] + ptsIntegral[idRightTop]);
                double sumGray = integral[idRightBot] + integral[idLeftTop] - (integral[idLeftBot] + integral[idRightTop]);
                if (ptsCnt <= 0)
                {
                    continue;
                }
                data[id2] = float(sumGray / ptsCnt);
            }
        }
        int s = wnd / 2 * 2 + 1;
        if (s > 201)
        {
            s = 201;
        }
        //GaussianBlur(depth, depth, Size(s, s), s, s);
    }
}

void gray2color(Mat img_gray, Mat img_color)
{
    Mat img = img_gray.clone();
    #define IMG_B(img,y,x) img.at<Vec3b>(y,x)[0]
    #define IMG_G(img,y,x) img.at<Vec3b>(y,x)[1]
    #define IMG_R(img,y,x) img.at<Vec3b>(y,x)[2]
     uchar tmp2=0;
     for (int y=0;y<img.rows;y++)//转为彩虹图的具体算法，主要思路是把灰度图对应的0～255的数值分别转换成彩虹色：红、橙、黄、绿、青、蓝。
      {
             for (int x=0;x<img.cols;x++)
             {
                    tmp2 = img.at<uchar>(y,x);
                    if (tmp2 <= 51)
                    {
                           IMG_B(img_color,y,x) = 255;
                           IMG_G(img_color,y,x) = tmp2*5;
                           IMG_R(img_color,y,x) = 0;
                    }
                    else if (tmp2 <= 102)
                    {
                           tmp2-=51;
                           IMG_B(img_color,y,x) = 255-tmp2*5;
                           IMG_G(img_color,y,x) = 255;
                           IMG_R(img_color,y,x) = 0;
                     }
                     else if (tmp2 <= 153)
                     {
                            tmp2-=102;
                            IMG_B(img_color,y,x) = 0;
                            IMG_G(img_color,y,x) = 255;
                            IMG_R(img_color,y,x) = tmp2*5;
                     }
                    else if (tmp2 <= 204)
                     {
                             tmp2-=153;
                             IMG_B(img_color,y,x) = 0;
                             IMG_G(img_color,y,x) = 255-uchar(128.0*tmp2/51.0+0.5);
                             IMG_R(img_color,y,x) = 255;
                     }
                     else
                      {
                             tmp2-=204;
                             IMG_B(img_color,y,x) = 0;
                             IMG_G(img_color,y,x) = 127-uchar(127.0*tmp2/51.0+0.5);
                             IMG_R(img_color,y,x) = 255;
                       }
                }
          }
}

void onMouse(int event, int x, int y,int,void*)
{
    cv::Point origin;
    switch (event)
    {
        case cv::EVENT_LBUTTONDOWN:   //鼠标左按钮按下的事件
            origin = cv::Point(x, y);
            //xyz.at<cv::Vec3f>(origin)[2] +=2;
            //std::cout << origin << "in world coordinate is: " << xyz.at<cv::Vec3f>(origin)<< std::endl;
            cout << origin << "depth is : " << depthMap.at<float>(origin)<< endl;
            break;
    }
}