#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/ocl.hpp"

#include "stitching.hpp"

#include <thread>
#include <iostream>
#include <atomic>

#include "atomicops.h"
#include "readerwriterqueue.h"

#include "debug_print.h"

using namespace std;
using namespace cv;
using namespace moodycamel;

// Options for threading 
#define NUMBER_OF_GPU 4
#define NUMBER_OF_THREAD NUMBER_OF_GPU * 8
#define QUEUE_SIZE 500
#define TEST_COUNT NUMBER_OF_THREAD * 10

// Option for CUDA implementation
bool try_gpu = true;

// Quit flag for threads
atomic_flag is_quit[NUMBER_OF_THREAD] = {ATOMIC_FLAG_INIT, };

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
    START_TIME(Create_stitcher);
    Mat output;
    Ptr<Stitcher_mod> stitcher = Stitcher_mod::create(Stitcher_mod::PANORAMA, try_gpu);
    STOP_TIME(Create_stitcher);

    START_TIME(Setup_stitcher_params);
	stitcher->setRegistrationResol(0.5);
	stitcher->setSeamEstimationResol(0.1);
	stitcher->setCompositingResol(Stitcher_mod::ORIG_RESOL);
	stitcher->setPanoConfidenceThresh(1.0);
	stitcher->setWaveCorrection(true);
	stitcher->setWaveCorrectKind(detail::WAVE_CORRECT_HORIZ);
    STOP_TIME(Setup_stitcher_params);

    START_TIME(Setup_stitcher_modules);
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
    STOP_TIME(Setup_stitcher_modules);

    START_TIME(Stitch_Time);
    Stitcher_mod::Status status = stitcher->stitch(imgs, output);
    STOP_TIME(Stitch_Time);

    return output;
}

// Stitcher thread
// wait input queue, stitch, push to output queue
void stitcher_thread(int idx)
{
	if (try_gpu)
	{
		int idx_modula = idx % NUMBER_OF_GPU;
		ocl::setUseOpenCL(false);
		cuda::setDevice(idx_modula);
	}
    // check quit flag
    while(is_quit[idx].test_and_set())
    {
        thread_args th_arg_t;

        //check input queue with timeout
        if(th_arg[idx].wait_dequeue_timed(th_arg_t, chrono::milliseconds(5)))
        {
            DEBUG_PRINT_OUT("Thread number: " << idx << " start stitching");
            thread_output th_out_t;
            th_out_t.pano = stitching(th_arg_t.imgs);

            DEBUG_PRINT_OUT("Thread number: " << idx << " push output to queue");
            th_out[idx].enqueue(th_out_t);
        }
    }
    DEBUG_PRINT_OUT("Thread number: " << idx << " quit");
}

// main thread
// prepare video, create threads and queues, put frames to queue and wait output, show it
int main(int argc, char* argv[])
{
	START_TIME(Total_Stitch_time);

    // read videos
    DEBUG_PRINT_OUT("start");
    VideoCapture vid0("videofile0.avi");
    VideoCapture vid1("videofile1.avi");
    VideoCapture vid2("videofile2.avi");

    vector<Mat> output;

    // create threads and quit flags
    thread stitch_thread[NUMBER_OF_THREAD];
    for(int i = 0; i < NUMBER_OF_THREAD; i++)
    {
        is_quit[i].test_and_set();
        stitch_thread[i] = thread(stitcher_thread, i);
    }

    // first stitching will do in main thread
    vector<Mat> vids(3);
        
    vid0 >> vids[0];
    vid1 >> vids[1];
    vid2 >> vids[2];

    DEBUG_PRINT_OUT("start stitching first frame");
    Mat pano = stitching(vids);

    DEBUG_PRINT_OUT("push to output queue");
    output.push_back(pano);

    // After first frame stitching done, next frames will be stitched with multi-thread 
    // read frames and push it to thread input queues
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

    // wait for threads output queue
    for(int i = 0; i < capture_count; i++)
    {
        thread_output th_out_t;

        th_out[i % 4].wait_dequeue(th_out_t);

        output.push_back(th_out_t.pano);
    }

    // It seams all done, quit all threads
    for(int i = 0; i < NUMBER_OF_THREAD; i++)
    {
        is_quit[i].clear();
        stitch_thread[i].join();
    }
    DEBUG_PRINT_OUT("All stitch threads join\n");

    // check outputs of stitcher
    for(int i = 0; i < output.size(); i++)
    {
        Mat result;
        output[i].convertTo(result, CV_8UC1);
        resize(result, result, output[0].size());
        imshow("stitch output", result);
        waitKey(0);
    }

    DEBUG_PRINT_OUT("stitching completed successfully\n");
	STOP_TIME(Total_Stitch_time);
	DEBUG_PRINT_OUT("Stitched frames : " << output.size());
	DEBUG_PRINT_OUT("Stitched Frames per sec : " << (output.size()*1.0f) / (Total_Stitch_time/getTickFrequency()*1.0f));
    return 0;
}