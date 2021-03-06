#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/core/ocl.hpp"

#include "stitching.hpp"

#include <thread>
#include <iostream>
#include <atomic>
#include <mutex>
#if STITCHER_DEBUG_IMWRTIE == true
#include <sstream>
#endif

#include "atomicops.h"
#include "readerwriterqueue.h"

#include "debug_print.h"

using namespace std;
using namespace cv;
using namespace moodycamel;

// Options for threading 
#define NUMBER_OF_GPU 4
#define NUMBER_OF_THREAD (NUMBER_OF_GPU * 1)
#define QUEUE_SIZE 500
#define TEST_COUNT (NUMBER_OF_THREAD * 3)

// Option for CUDA implementation
bool try_gpu = true;
bool use_ocl = true;

// USE stitcher's default setting
bool stitcher_default = false;

// Option for Optical flow motion compensation
#define USE_OPT_FLOW true

// Option for video output
#define VIDEO_OUT true
#if VIDEO_OUT == true
static VideoWriter pano_out;
#define VIDEO_OUT_NAME "pano_out.avi"
#define VIDEO_OUT_FPS 30
#endif

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
#if STITCHER_DEBUG_IMWRTIE == true
	vector<Mat> debug_mat;
#endif
};

// Stitcher thread's input queue structure queue
BlockingReaderWriterQueue<thread_args>      th_arg[NUMBER_OF_THREAD];
// Stitcher thread's output queue structure queue
BlockingReaderWriterQueue<thread_output>    th_out[NUMBER_OF_THREAD];

#if USE_OPT_FLOW == true
mutex first_frames_mutex;

vector<Mat> first_frames;
vector<detail::MatchesInfo> first_match_infos;
vector<detail::CameraParams> first_cameras;
#endif

//OpenCV Mat handle util
void clear_vector_mat(vector<Mat> &vec_mat)
{
    for(int i = 0; i < vec_mat.size(); i++)
        vec_mat[i].release();
    vec_mat = vector<Mat>();
}

void deep_copy_vector_mat(vector<Mat> &src, vector<Mat> &dst)
{
    clear_vector_mat(dst);
    for(int i = 0; i < src.size(); i++)
        dst.push_back(src[i].clone());
}

Ptr<Stitcher_mod> stitcher_setup()
{
    START_TIME(Create_stitcher);
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
	if (stitcher_default)
	{
		stitcher->setFeaturesFinder(makePtr<detail::OrbFeaturesFinder>());
		stitcher->setWarper(makePtr<SphericalWarper>());
		stitcher->setSeamFinder(makePtr<detail::GraphCutSeamFinder>(detail::GraphCutSeamFinderBase::COST_COLOR));
		stitcher->setExposureCompensator(makePtr<detail::BlocksGainCompensator>());
		stitcher->setBlender(makePtr<detail::MultiBandBlender>(false));
		stitcher->setFeaturesMatcher(makePtr<detail::BestOf2NearestMatcher>(false));
		stitcher->setBundleAdjuster(makePtr<detail::BundleAdjusterRay>());
	}
	else
	{
		if (try_gpu)
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
	}
    STOP_TIME(Setup_stitcher_modules);

    return stitcher;
}

#if USE_OPT_FLOW == true
// Stitcher module implementation
// create stitcher instance, setup, stitch frames
// also prepare for next optical flow stitching 
Mat stitching_first_frame(vector<Mat> &first_frames, vector<detail::CameraParams> &first_cameras, vector<detail::MatchesInfo> &first_match_infos)
{
    Ptr<Stitcher_mod> stitcher = stitcher_setup();

    START_TIME(Stitch_Time);
    Mat output;
    Stitcher_mod::Status status = stitcher->estimateTransform(first_frames);

    first_cameras = stitcher->cameras();
    first_match_infos = stitcher->matchInfo();

    stitcher->composePanorama(output);
    STOP_TIME(Stitch_Time);

    return output;
}

// Stitcher module implementation with Optical flow
// create stitcher instance, setup, stitch frames
Mat stitching_optical_flow(vector<Mat> &prev_frames, vector<detail::MatchesInfo> &prev_match_infos, vector<detail::CameraParams> &prev_cameras
                            , vector<Mat> &current_frames)
{
    vector<Mat> prev_frames_, current_frames_;
    deep_copy_vector_mat(prev_frames, prev_frames_);
    deep_copy_vector_mat(current_frames, current_frames_);

    Ptr<Stitcher_mod> stitcher = stitcher_setup();

    START_TIME(Stitch_Time);
    Mat output;
    Stitcher_mod::Status status = stitcher->estimateTransformOpticalFlow(prev_frames_, prev_match_infos, prev_cameras
                                                                        , current_frames_, std::vector<std::vector<Rect> >());

    deep_copy_vector_mat(current_frames, prev_frames);
    prev_cameras = stitcher->cameras();
    prev_match_infos = stitcher->matchInfo();

    stitcher->composePanorama(output);
    STOP_TIME(Stitch_Time);

    return output;
}
#else
// Stitcher module implementation
// create stitcher instance, setup, stitch frames
Mat stitching(vector<Mat> &imgs)
{
    Ptr<Stitcher_mod> stitcher = stitcher_setup();

    START_TIME(Stitch_Time);
    Mat output;
    Stitcher_mod::Status status = stitcher->stitch(imgs, output);
    STOP_TIME(Stitch_Time);

    return output;
}
#endif

#if STITCHER_DEBUG_IMWRTIE == true
//return debug matrix with output panorama
Mat stitching(vector<Mat> &imgs, vector<Mat> &debug_mat)
{
    Ptr<Stitcher_mod> stitcher = stitcher_setup();

	START_TIME(Stitch_Time);
    Mat output;
	Stitcher_mod::Status status = stitcher->stitch(imgs, output);
	STOP_TIME(Stitch_Time);

	vector<Mat> tmp = stitcher->debugMat();
	for (int i = 0; i < tmp.size(); i++)
	{
		debug_mat.push_back(tmp[i].clone());
	}

	return output;
}
#endif

// Stitcher thread
// wait input queue, stitch, push to output queue
void stitcher_thread(int idx)
{
	if (try_gpu)
	{
		int idx_modula = idx % NUMBER_OF_GPU;
		cuda::setDevice(idx_modula);
	}

#if USE_OPT_FLOW == true
    bool is_first = true;
    vector<Mat> prev_frames;
    vector<detail::CameraParams> prev_cameras;
    vector<detail::MatchesInfo> prev_match_infos;
#endif

    // check quit flag
    while(is_quit[idx].test_and_set())
    {
        thread_args th_arg_t;

        //check input queue with timeout
        if(th_arg[idx].wait_dequeue_timed(th_arg_t, chrono::milliseconds(5)))
        {
            DEBUG_PRINT_OUT("Thread number: " << idx << " start stitching");
            thread_output th_out_t;
#if USE_OPT_FLOW == true
            if(is_first)
            {
                first_frames_mutex.lock();
                deep_copy_vector_mat(first_frames, prev_frames);
                prev_cameras = first_cameras;
                prev_match_infos = first_match_infos;
                first_frames_mutex.unlock();

                is_first = false;
            }
            th_out_t.pano = stitching_optical_flow(prev_frames, prev_match_infos, prev_cameras, th_arg_t.imgs);
#else
#if STITCHER_DEBUG_IMWRTIE == true
			th_out_t.pano = stitching(th_arg_t.imgs, th_out_t.debug_mat);
#else
			th_out_t.pano = stitching(th_arg_t.imgs);
#endif
#endif

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
	if (use_ocl)
		ocl::setUseOpenCL(true);
	else
		ocl::setUseOpenCL(false);

    // read videos
    APP_PRINT_OUT("start");
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

	START_APP_TIME(Total_Stitch_time);

    DEBUG_PRINT_OUT("start stitching first frame");
#if USE_OPT_FLOW == true
    first_frames_mutex.lock();
    deep_copy_vector_mat(vids, first_frames);
    Mat pano = stitching_first_frame(first_frames, first_cameras, first_match_infos);
    first_frames_mutex.unlock();
#else
    Mat pano = stitching(vids);
#endif

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
        th_arg_t.imgs.push_back(tmp.clone());
        vid1 >> tmp;
        th_arg_t.imgs.push_back(tmp.clone());
        vid2 >> tmp;
        th_arg_t.imgs.push_back(tmp.clone());

        th_arg[capture_count % NUMBER_OF_THREAD].enqueue(th_arg_t);

        capture_count++;
    }

    // wait for threads output queue
    for(int i = 0; i < capture_count; i++)
    {
        thread_output th_out_t;

        th_out[i % NUMBER_OF_THREAD].wait_dequeue(th_out_t);

        output.push_back(th_out_t.pano);
#if STITCHER_DEBUG_IMWRTIE == true
		stringstream debug_name;
		debug_name << "debug " << i << 0 << ".jpg";
		imwrite(debug_name.str(), th_out_t.debug_mat[0]);
		debug_name.str("");

		debug_name << "debug " << i << 1 << ".jpg";
		imwrite(debug_name.str(), th_out_t.debug_mat[1]);
		debug_name.str("");

		debug_name << "debug " << i << 2 << ".jpg";
		imwrite(debug_name.str(), th_out_t.debug_mat[2]);
		debug_name.str("");
#endif
    }

    // It seams all done, quit all threads
    for(int i = 0; i < NUMBER_OF_THREAD; i++)
    {
        is_quit[i].clear();
        stitch_thread[i].join();
    }
    DEBUG_PRINT_OUT("All stitch threads join\n");

	APP_PRINT_OUT("stitching completed successfully\n");
	STOP_APP_TIME(Total_Stitch_time);
	APP_PRINT_OUT("Stitched frames : " << output.size());
	APP_PRINT_OUT("Stitched Frames per sec : " << (output.size()*1.0f) / (Total_Stitch_time / getTickFrequency()*1.0f));

#if VIDEO_OUT == true
	//prepare output video
	Size first_frame_size = output[0].size();
	pano_out.open(VIDEO_OUT_NAME, CV_FOURCC('M', 'P', '4', '2'), VIDEO_OUT_FPS, first_frame_size, true);

	for (int i = 0; i < output.size(); i++)
	{
		Mat result;
		output[i].convertTo(result, CV_8UC1);
		resize(result, result, first_frame_size);

		pano_out << result;
	}
#else
    // check outputs of stitcher
    for(int i = 0; i < output.size(); i++)
    {
        Mat result;
        output[i].convertTo(result, CV_8UC1);
        resize(result, result, output[0].size());
        imshow("stitch output", result);
        waitKey(0);
    }
#endif
    return 0;
}