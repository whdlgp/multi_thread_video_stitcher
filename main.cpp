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

#define NUMBER_OF_THREAD 4
#define QUEUE_SIZE 500
#define TEST_COUNT 4
bool try_gpu = false;

// Stitcher thread's input queue structure 
class thread_args
{
    public :
    vector<Mat> imgs;
};

// Stitcher thread's output queue structure
class thread_output
{
    public :
    Mat pano;
};

// Stitcher thread's input queue structure queue
BlockingReaderWriterQueue<thread_args>      th_arg[NUMBER_OF_THREAD];
// Stitcher thread's output queue structure queue
BlockingReaderWriterQueue<thread_output>    th_out[NUMBER_OF_THREAD];

// Stitcher module implementation
// create stitcher instance, setup, stitch frames
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
		#if defined(HAVE_OPENCV_XFEATURES2D) && defined(HAVE_OPENCV_CUDALEGACY)
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

// Stitcher thread
// wait input queue, stitch, push to output queue
void stitcher_thread(int idx)
{
	if (try_gpu)
	{
		ocl::setUseOpenCL(false);
		cuda::setDevice(idx);
	}
    while(true)
    {
        thread_args th_arg_t;

        th_arg[idx].wait_dequeue(th_arg_t);

        DEBUG_PRINT_OUT("Thread number: " << idx << " start stitching");
        thread_output th_out_t;
        th_out_t.pano = stitching(th_arg_t.imgs);

        DEBUG_PRINT_OUT("Thread number: " << idx << " push output to queue");
        th_out[idx].enqueue(th_out_t);
    }
}

// main thread
// prepare video, create threads and queues, put frames to queue and wait output, show it
int main(int argc, char* argv[])
{
    DEBUG_PRINT_OUT("start");
    VideoCapture vid0("videofile0.avi");
    VideoCapture vid1("videofile1.avi");
    VideoCapture vid2("videofile2.avi");

    vector<Mat> output;

    thread stitch_thread[NUMBER_OF_THREAD];
    for(int i = 0; i < NUMBER_OF_THREAD; i++)
        stitch_thread[i] = thread(stitcher_thread, i);

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
        
        thread_args th_arg_t;
        Mat tmp;
        vid0 >> tmp;
        th_arg_t.imgs.push_back(tmp);
        vid1 >> tmp;
        th_arg_t.imgs.push_back(tmp);
        vid2 >> tmp;
        th_arg_t.imgs.push_back(tmp);

        th_arg[capture_count % 4].enqueue(th_arg_t);

        capture_count++;
    }

    for(int i = 0; i < capture_count; i++)
    {
        thread_output th_out_t;

        th_out[i % 4].wait_dequeue(th_out_t);

        output.push_back(th_out_t.pano);
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