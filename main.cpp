#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/core/ocl.hpp"

#include <thread>
#include <iostream>

#include "atomicops.h"
#include "readerwriterqueue.h"

using namespace std;
using namespace cv;
using namespace moodycamel;

#define DEBUG_PRINT

#ifdef DEBUG_PRINT
#define DEBUG_PRINT_ERR(x) (std::cerr << x << std::endl)
#define DEBUG_PRINT_OUT(x) (std::cout << x << std::endl)
#else
#define DEBUG_PRINT_ERR(x)
#define DEBUG_PRINT_OUT(x)
#endif

#define QUEUE_SIZE 500
#define TEST_COUNT 4
bool try_gpu = false;

class thread_args
{
    public :
    vector<Mat> imgs;
};

class thread_output
{
    public :
    Mat pano;
};

BlockingReaderWriterQueue<thread_args> th_arg0(QUEUE_SIZE);
BlockingReaderWriterQueue<thread_args> th_arg1(QUEUE_SIZE);
BlockingReaderWriterQueue<thread_args> th_arg2(QUEUE_SIZE);
BlockingReaderWriterQueue<thread_args> th_arg3(QUEUE_SIZE);

BlockingReaderWriterQueue<thread_output> th_out0(QUEUE_SIZE);
BlockingReaderWriterQueue<thread_output> th_out1(QUEUE_SIZE);
BlockingReaderWriterQueue<thread_output> th_out2(QUEUE_SIZE);
BlockingReaderWriterQueue<thread_output> th_out3(QUEUE_SIZE);

Mat stitching(vector<Mat> imgs)
{
    Mat output;
    Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::PANORAMA, try_gpu);

	stitcher->setRegistrationResol(0.5);
	stitcher->setSeamEstimationResol(0.1);
	stitcher->setCompositingResol(Stitcher::ORIG_RESOL);
	stitcher->setPanoConfidenceThresh(1.0);
	stitcher->setWaveCorrection(true);
	stitcher->setWaveCorrectKind(detail::WAVE_CORRECT_HORIZ);

    if(try_gpu)
    {
		#ifdef HAVE_OPENCV_XFEATURES2D
		stitcher->setFeaturesFinder(makePtr<detail::SurfFeaturesFinderGpu>());		//GPU
		#endif
		stitcher->setFeaturesMatcher(makePtr<detail::BestOf2NearestMatcher>(true));
		stitcher->setBundleAdjuster(makePtr<detail::BundleAdjusterRay>());
		#ifdef HAVE_OPENCV_CUDAWARPING
		stitcher->setWarper(makePtr<SphericalWarperGpu>());							//GPU
		#endif	
		stitcher->setExposureCompensator(makePtr<detail::BlocksGainCompensator>());
		stitcher->setSeamFinder(makePtr<detail::VoronoiSeamFinder>());
		#if defined(HAVE_OPENCV_CUDAARITHM) && defined(HAVE_OPENCV_CUDAWARPING)
		stitcher->setBlender(makePtr<detail::MultiBandBlender>(true));				//GPU
		#endif	
    }
    else
    {
        stitcher->setFeaturesFinder(makePtr<detail::OrbFeaturesFinder>());
        stitcher->setFeaturesMatcher(makePtr<detail::BestOf2NearestMatcher>(false));
        stitcher->setBundleAdjuster(makePtr<detail::BundleAdjusterRay>());
        stitcher->setWarper(makePtr<SphericalWarper>());
        stitcher->setExposureCompensator(makePtr<detail::BlocksGainCompensator>());
        stitcher->setSeamFinder(makePtr<detail::VoronoiSeamFinder>());
		stitcher->setBlender(makePtr<detail::MultiBandBlender>(false));
    }

    Stitcher::Status status = stitcher->stitch(imgs, output);

    return output;
}

void stitcher_thread(int idx)
{
	if (try_gpu)
	{
		ocl::setUseOpenCL(false);
		cuda::setDevice(idx);
	}
    while(true)
    {
        thread_args th_arg;

        switch(idx)
        {
            case 0:
            th_arg0.wait_dequeue(th_arg);
            break;

            case 1:
            th_arg1.wait_dequeue(th_arg);
            break;

            case 2:
            th_arg2.wait_dequeue(th_arg);
            break;

            case 3:
            th_arg3.wait_dequeue(th_arg);
            break;
        }

        DEBUG_PRINT_OUT("Thread number: " << idx << " start stitching");
        thread_output th_out;
        th_out.pano = stitching(th_arg.imgs);

        DEBUG_PRINT_OUT("Thread number: " << idx << " push output to queue");
        switch(idx)
        {
            case 0:
            th_out0.enqueue(th_out);
            break;

            case 1:
            th_out1.enqueue(th_out);
            break;
            
            case 2:
            th_out2.enqueue(th_out);
            break;
            
            case 3:
            th_out3.enqueue(th_out);
            break;
        }
    }
}

int main(int argc, char* argv[])
{
    DEBUG_PRINT_OUT("start");
    VideoCapture vid0("videofile0.avi");
    VideoCapture vid1("videofile1.avi");
    VideoCapture vid2("videofile2.avi");

    vector<Mat> output;

    thread thread0(stitcher_thread, 0);
    thread thread1(stitcher_thread, 1);
    thread thread2(stitcher_thread, 2);
    thread thread3(stitcher_thread, 3);

    vector<Mat> vids(3);
        
    vid0 >> vids[0];
    vid1 >> vids[1];
    vid2 >> vids[2];

    DEBUG_PRINT_OUT("start stitching first frame");
    Mat pano = stitching(vids);

    DEBUG_PRINT_OUT("push to output queue");
    output.push_back(pano);

    int capture_count = 0;
    while(capture_count < TEST_COUNT)
    {
        DEBUG_PRINT_OUT("Put frame");
        
        vector<Mat> vids(3);
        
        thread_args th_arg;
        Mat tmp;
        vid0 >> tmp;
        th_arg.imgs.push_back(tmp);
        vid1 >> tmp;
        th_arg.imgs.push_back(tmp);
        vid2 >> tmp;
        th_arg.imgs.push_back(tmp);

        switch(capture_count % 4)
        {
            case 0:
            th_arg0.enqueue(th_arg);
            break;

            case 1:
            th_arg1.enqueue(th_arg);
            break;
            
            case 2:
            th_arg2.enqueue(th_arg);
            break;
            
            case 3:
            th_arg3.enqueue(th_arg);
            break;
        }

        capture_count++;
    }

    for(int i = 0; i < capture_count; i++)
    {
        thread_output th_out;
        switch(i % 4)
        {
            case 0:
            th_out0.wait_dequeue(th_out);
            break;

            case 1:
            th_out1.wait_dequeue(th_out);
            break;
            
            case 2:
            th_out2.wait_dequeue(th_out);
            break;
            
            case 3:
            th_out3.wait_dequeue(th_out);
            break;
        }

        output.push_back(th_out.pano);
    }

    for(int i = 0; i < output.size(); i++)
    {
        Mat result;
        output[i].convertTo(result, CV_8UC1);
        resize(result, result, Size(1280, 240));
        imshow("stitch output", output[i]);
        waitKey(0);
    }

    DEBUG_PRINT_OUT("stitching completed successfully\n");
    return 0;
}