/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "opencv2/opencv_modules.hpp"

#include <vector>
#include <algorithm>
#include <utility>
#include <set>
#include <functional>
#include <sstream>
#include <iostream>
#include <cmath>
#include "opencv2/core.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/core/utility.hpp"
#include "stitching.hpp"
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/timelapsers.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"

#ifdef HAVE_OPENCV_CUDAARITHM
#  include "opencv2/cudaarithm.hpp"
#endif

#ifdef HAVE_OPENCV_CUDAWARPING
#  include "opencv2/cudawarping.hpp"
#endif

#ifdef HAVE_OPENCV_CUDAFEATURES2D
#  include "opencv2/cudafeatures2d.hpp"
#endif

#ifdef HAVE_OPENCV_CUDALEGACY
#  include "opencv2/cudalegacy.hpp"
#endif

#ifdef HAVE_OPENCV_XFEATURES2D
#  include "opencv2/xfeatures2d/cuda.hpp"
#endif

#ifdef HAVE_TEGRA_OPTIMIZATION
# include "opencv2/stitching/stitching_tegra.hpp"
#endif

#include "debug_print.h"

namespace cv {

Stitcher_mod Stitcher_mod::createDefault(bool try_use_gpu)
{
    Stitcher_mod stitcher;
    stitcher.setRegistrationResol(0.6);
    stitcher.setSeamEstimationResol(0.1);
    stitcher.setCompositingResol(ORIG_RESOL);
    stitcher.setPanoConfidenceThresh(1);
    stitcher.setWaveCorrection(true);
    stitcher.setWaveCorrectKind(detail::WAVE_CORRECT_HORIZ);
    stitcher.setFeaturesMatcher(makePtr<detail::BestOf2NearestMatcher>(try_use_gpu));
    stitcher.setBundleAdjuster(makePtr<detail::BundleAdjusterRay>());

#ifdef HAVE_OPENCV_CUDALEGACY
    if (try_use_gpu && cuda::getCudaEnabledDeviceCount() > 0)
    {
#ifdef HAVE_OPENCV_XFEATURES2D
        stitcher.setFeaturesFinder(makePtr<detail::SurfFeaturesFinderGpu>());
#else
        stitcher.setFeaturesFinder(makePtr<detail::OrbFeaturesFinder>());
#endif
        stitcher.setWarper(makePtr<SphericalWarperGpu>());
        stitcher.setSeamFinder(makePtr<detail::GraphCutSeamFinderGpu>());
    }
    else
#endif
    {
#ifdef HAVE_OPENCV_XFEATURES2D
        stitcher.setFeaturesFinder(makePtr<detail::SurfFeaturesFinder>());
#else
        stitcher.setFeaturesFinder(makePtr<detail::OrbFeaturesFinder>());
#endif
        stitcher.setWarper(makePtr<SphericalWarper>());
        stitcher.setSeamFinder(makePtr<detail::GraphCutSeamFinder>(detail::GraphCutSeamFinderBase::COST_COLOR));
    }

    stitcher.setExposureCompensator(makePtr<detail::BlocksGainCompensator>());
    stitcher.setBlender(makePtr<detail::MultiBandBlender>(try_use_gpu));

    stitcher.work_scale_ = 1;
    stitcher.seam_scale_ = 1;
    stitcher.seam_work_aspect_ = 1;
    stitcher.warped_image_scale_ = 1;

    return stitcher;
}


Ptr<Stitcher_mod> Stitcher_mod::create(Mode mode, bool try_use_gpu)
{
    Stitcher_mod stit = createDefault(try_use_gpu);
    Ptr<Stitcher_mod> stitcher = makePtr<Stitcher_mod>(stit);

    switch (mode)
    {
    case PANORAMA: // PANORAMA is the default
        // already setup
    break;

    case SCANS:
        stitcher->setWaveCorrection(false);
        stitcher->setFeaturesMatcher(makePtr<detail::AffineBestOf2NearestMatcher>(false, try_use_gpu));
        stitcher->setBundleAdjuster(makePtr<detail::BundleAdjusterAffinePartial>());
        stitcher->setWarper(makePtr<AffineWarper>());
        stitcher->setExposureCompensator(makePtr<detail::NoExposureCompensator>());
    break;

    default:
        CV_Error(Error::StsBadArg, "Invalid stitching mode. Must be one of Stitcher_mod::Mode");
    break;
    }

    return stitcher;
}


Stitcher_mod::Status Stitcher_mod::estimateTransform(InputArrayOfArrays images)
{
    return estimateTransform(images, std::vector<std::vector<Rect> >());
}


Stitcher_mod::Status Stitcher_mod::estimateTransform(InputArrayOfArrays images, const std::vector<std::vector<Rect> > &rois)
{
    images.getUMatVector(imgs_);
    rois_ = rois;

    Status status;

    if ((status = matchImages()) != OK)
        return status;

    if ((status = estimateCameraParams()) != OK)
        return status;

    return OK;
}



Stitcher_mod::Status Stitcher_mod::composePanorama(OutputArray pano)
{
    return composePanorama(std::vector<UMat>(), pano);
}


Stitcher_mod::Status Stitcher_mod::composePanorama(InputArrayOfArrays images, OutputArray pano)
{
    std::vector<UMat> imgs;
    images.getUMatVector(imgs);
    if (!imgs.empty())
    {
        CV_Assert(imgs.size() == imgs_.size());

        UMat img;
        seam_est_imgs_.resize(imgs.size());

        START_TIME(Resize_in_seam_scale);
        for (size_t i = 0; i < imgs.size(); ++i)
        {
            imgs_[i] = imgs[i];
            resize(imgs[i], img, Size(), seam_scale_, seam_scale_, INTER_LINEAR_EXACT);
            seam_est_imgs_[i] = img.clone();
        }
        STOP_TIME(Resize_in_seam_scale);

        std::vector<UMat> seam_est_imgs_subset;
        std::vector<UMat> imgs_subset;

        for (size_t i = 0; i < indices_.size(); ++i)
        {
            imgs_subset.push_back(imgs_[indices_[i]]);
            seam_est_imgs_subset.push_back(seam_est_imgs_[indices_[i]]);
        }

        seam_est_imgs_ = seam_est_imgs_subset;
        imgs_ = imgs_subset;
    }

    UMat pano_;

#if ENABLE_LOG
    int64 t = getTickCount();
#endif

    std::vector<Point> corners(imgs_.size());
    std::vector<UMat> masks_warped(imgs_.size());
    std::vector<UMat> images_warped(imgs_.size());
    std::vector<Size> sizes(imgs_.size());
    std::vector<UMat> masks(imgs_.size());

    // Prepare image masks
    for (size_t i = 0; i < imgs_.size(); ++i)
    {
        masks[i].create(seam_est_imgs_[i].size(), CV_8U);
        masks[i].setTo(Scalar::all(255));
    }

    START_TIME(Warping_Image_and_Mask_in_seam_scale);
    // Warp images and their masks
    Ptr<detail::RotationWarper> w = warper_->create(float(warped_image_scale_ * seam_work_aspect_));
    for (size_t i = 0; i < imgs_.size(); ++i)
    {
        Mat_<float> K;
        cameras_[i].K().convertTo(K, CV_32F);
        K(0,0) *= (float)seam_work_aspect_;
        K(0,2) *= (float)seam_work_aspect_;
        K(1,1) *= (float)seam_work_aspect_;
        K(1,2) *= (float)seam_work_aspect_;

        corners[i] = w->warp(seam_est_imgs_[i], K, cameras_[i].R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
        sizes[i] = images_warped[i].size();

        w->warp(masks[i], K, cameras_[i].R, INTER_NEAREST, BORDER_CONSTANT, masks_warped[i]);
    }
    STOP_TIME(Warping_Image_and_Mask_in_seam_scale);

    START_TIME(Feeding_exposure_compensation_in_seam_scale);
    // Compensate exposure before finding seams
    exposure_comp_->feed(corners, images_warped, masks_warped);
    STOP_TIME(Feeding_exposure_compensation_in_seam_scale);
    START_TIME(Applying_exposure_compensation_in_seam_scale);
    for (size_t i = 0; i < imgs_.size(); ++i)
        exposure_comp_->apply(int(i), corners[i], images_warped[i], masks_warped[i]);
    STOP_TIME(Applying_exposure_compensation_in_seam_scale);

    START_TIME(Finding_seam_in_seam_scale);
    // Find seams
    std::vector<UMat> images_warped_f(imgs_.size());
    for (size_t i = 0; i < imgs_.size(); ++i)
        images_warped[i].convertTo(images_warped_f[i], CV_32F);
    seam_finder_->find(images_warped_f, corners, masks_warped);
    STOP_TIME(Finding_seam_in_seam_scale);

    // Release unused memory
    seam_est_imgs_.clear();
    images_warped.clear();
    images_warped_f.clear();
    masks.clear();

#if ENABLE_LOG
    t = getTickCount();
#endif

    UMat img_warped, img_warped_s;
    UMat dilated_mask, seam_mask, mask, mask_warped;

    //double compose_seam_aspect = 1;
    double compose_work_aspect = 1;
    bool is_blender_prepared = false;

    double compose_scale = 1;
    bool is_compose_scale_set = false;

    std::vector<detail::CameraParams> cameras_scaled(cameras_);

    UMat full_img, img;
    for (size_t img_idx = 0; img_idx < imgs_.size(); ++img_idx)
    {
        // Read image and resize it if necessary
        full_img = imgs_[img_idx];
        if (!is_compose_scale_set)
        {
            if (compose_resol_ > 0)
                compose_scale = std::min(1.0, std::sqrt(compose_resol_ * 1e6 / full_img.size().area()));
            is_compose_scale_set = true;

            // Compute relative scales
            //compose_seam_aspect = compose_scale / seam_scale_;
            compose_work_aspect = compose_scale / work_scale_;

            // Update warped image scale
            float warp_scale = static_cast<float>(warped_image_scale_ * compose_work_aspect);
            w = warper_->create(warp_scale);

            // Update corners and sizes
            for (size_t i = 0; i < imgs_.size(); ++i)
            {
                // Update intrinsics
                cameras_scaled[i].ppx *= compose_work_aspect;
                cameras_scaled[i].ppy *= compose_work_aspect;
                cameras_scaled[i].focal *= compose_work_aspect;

                // Update corner and size
                Size sz = full_img_sizes_[i];
                if (std::abs(compose_scale - 1) > 1e-1)
                {
                    sz.width = cvRound(full_img_sizes_[i].width * compose_scale);
                    sz.height = cvRound(full_img_sizes_[i].height * compose_scale);
                }

                Mat K;
                cameras_scaled[i].K().convertTo(K, CV_32F);
                Rect roi = w->warpRoi(sz, K, cameras_scaled[i].R);
                corners[i] = roi.tl();
                sizes[i] = roi.size();
            }
        }
        if (std::abs(compose_scale - 1) > 1e-1)
        {
            START_TIME(Resize_in_compose_scale);
            resize(full_img, img, Size(), compose_scale, compose_scale, INTER_LINEAR_EXACT);
            STOP_TIME(Resize_in_compose_scale);
        }
        else
            img = full_img;
        full_img.release();
        Size img_size = img.size();

        Mat K;
        cameras_scaled[img_idx].K().convertTo(K, CV_32F);

        START_TIME(Warping_Image_and_Mask_in_compose_scale);
        // Warp the current image
        w->warp(img, K, cameras_[img_idx].R, INTER_LINEAR, BORDER_REFLECT, img_warped);

#if STITCHER_DEBUG_IMWRTIE == true
		Mat img_mat;
		img_warped.copyTo(img_mat);
		debug_mat.push_back(img_mat);
#endif

        // Warp the current image mask
        mask.create(img_size, CV_8U);
        mask.setTo(Scalar::all(255));
        w->warp(mask, K, cameras_[img_idx].R, INTER_NEAREST, BORDER_CONSTANT, mask_warped);
        STOP_TIME(Warping_Image_and_Mask_in_compose_scale);

        START_TIME(Applying_exposure_compensation_in_compose_scale);
        // Compensate exposure
        exposure_comp_->apply((int)img_idx, corners[img_idx], img_warped, mask_warped);
        STOP_TIME(Applying_exposure_compensation_in_compose_scale);

        img_warped.convertTo(img_warped_s, CV_16S);
        img_warped.release();
        img.release();
        mask.release();

        START_TIME(Resize_seam_mask_to_compose_size_mask);
        // Make sure seam mask has proper size
        dilate(masks_warped[img_idx], dilated_mask, Mat());
        resize(dilated_mask, seam_mask, mask_warped.size(), 0, 0, INTER_LINEAR_EXACT);

        bitwise_and(seam_mask, mask_warped, mask_warped);
        STOP_TIME(Resize_seam_mask_to_compose_size_mask);

        START_TIME(Prepare_Blending);
        if (!is_blender_prepared)
        {
            blender_->prepare(corners, sizes);
            is_blender_prepared = true;
        }
        STOP_TIME(Prepare_Blending);

        START_TIME(Feeding_Blending);
        // Blend the current image
        blender_->feed(img_warped_s, mask_warped, corners[img_idx]);
        STOP_TIME(Feeding_Blending);
    }

    START_TIME(Blending);
    UMat result, result_mask;
    blender_->blend(result, result_mask);
    STOP_TIME(Blending);

    // Preliminary result is in CV_16SC3 format, but all values are in [0,255] range,
    // so convert it to avoid user confusing
    result.convertTo(pano, CV_8U);

    return OK;
}


Stitcher_mod::Status Stitcher_mod::stitch(InputArrayOfArrays images, OutputArray pano)
{
    Status status = estimateTransform(images);
    if (status != OK)
        return status;
    return composePanorama(pano);
}


Stitcher_mod::Status Stitcher_mod::stitch(InputArrayOfArrays images, const std::vector<std::vector<Rect> > &rois, OutputArray pano)
{
    Status status = estimateTransform(images, rois);
    if (status != OK)
        return status;
    return composePanorama(pano);
}


Stitcher_mod::Status Stitcher_mod::matchImages()
{
    if ((int)imgs_.size() < 2)
    {
        return ERR_NEED_MORE_IMGS;
    }

    work_scale_ = 1;
    seam_work_aspect_ = 1;
    seam_scale_ = 1;
    bool is_work_scale_set = false;
    bool is_seam_scale_set = false;
    UMat full_img, img;
    features_.resize(imgs_.size());
    seam_est_imgs_.resize(imgs_.size());
    full_img_sizes_.resize(imgs_.size());

    std::vector<UMat> feature_find_imgs(imgs_.size());
    std::vector<std::vector<Rect> > feature_find_rois(rois_.size());

    for (size_t i = 0; i < imgs_.size(); ++i)
    {
        full_img = imgs_[i];
        full_img_sizes_[i] = full_img.size();

        if (registr_resol_ < 0)
        {
            img = full_img;
            work_scale_ = 1;
            is_work_scale_set = true;
        }
        else
        {
            if (!is_work_scale_set)
            {
                work_scale_ = std::min(1.0, std::sqrt(registr_resol_ * 1e6 / full_img.size().area()));
                is_work_scale_set = true;
            }
            START_TIME(Resize_in_work_scale);
            resize(full_img, img, Size(), work_scale_, work_scale_, INTER_LINEAR_EXACT);
            STOP_TIME(Resize_in_work_scale);
        }
        if (!is_seam_scale_set)
        {
            seam_scale_ = std::min(1.0, std::sqrt(seam_est_resol_ * 1e6 / full_img.size().area()));
            seam_work_aspect_ = seam_scale_ / work_scale_;
            is_seam_scale_set = true;
        }

        if (rois_.empty())
            feature_find_imgs[i] = img;
        else
        {
            feature_find_rois[i].resize(rois_[i].size());
            for (size_t j = 0; j < rois_[i].size(); ++j)
            {
                Point tl(cvRound(rois_[i][j].x * work_scale_), cvRound(rois_[i][j].y * work_scale_));
                Point br(cvRound(rois_[i][j].br().x * work_scale_), cvRound(rois_[i][j].br().y * work_scale_));
                feature_find_rois[i][j] = Rect(tl, br);
            }
            feature_find_imgs[i] = img;
        }
        features_[i].img_idx = (int)i;

        START_TIME(Resize_in_seam_scale);
        resize(full_img, img, Size(), seam_scale_, seam_scale_, INTER_LINEAR_EXACT);
        seam_est_imgs_[i] = img.clone();
        STOP_TIME(Resize_in_seam_scale);
    }

    START_TIME(Finding_Feature_time);
    // find features possibly in parallel
    if (rois_.empty())
        (*features_finder_)(feature_find_imgs, features_);
    else
        (*features_finder_)(feature_find_imgs, features_, feature_find_rois);

    // Do it to save memory
    features_finder_->collectGarbage();
    full_img.release();
    img.release();
    feature_find_imgs.clear();
    feature_find_rois.clear();
    STOP_TIME(Finding_Feature_time);

    START_TIME(Matching_Feature_time);
    (*features_matcher_)(features_, pairwise_matches_, matching_mask_);
    features_matcher_->collectGarbage();
    STOP_TIME(Matching_Feature_time);

    // Leave only images we are sure are from the same panorama
    indices_ = detail::leaveBiggestComponent(features_, pairwise_matches_, (float)conf_thresh_);
    std::vector<UMat> seam_est_imgs_subset;
    std::vector<UMat> imgs_subset;
    std::vector<Size> full_img_sizes_subset;
    for (size_t i = 0; i < indices_.size(); ++i)
    {
        imgs_subset.push_back(imgs_[indices_[i]]);
        seam_est_imgs_subset.push_back(seam_est_imgs_[indices_[i]]);
        full_img_sizes_subset.push_back(full_img_sizes_[indices_[i]]);
    }
    seam_est_imgs_ = seam_est_imgs_subset;
    imgs_ = imgs_subset;
    full_img_sizes_ = full_img_sizes_subset;

    if ((int)imgs_.size() < 2)
    {
        return ERR_NEED_MORE_IMGS;
    }

    return OK;
}


Stitcher_mod::Status Stitcher_mod::estimateCameraParams()
{
    /* TODO OpenCV ABI 4.x
    get rid of this dynamic_cast hack and use estimator_
    */
    Ptr<detail::Estimator> estimator;
    if (dynamic_cast<detail::AffineBestOf2NearestMatcher*>(features_matcher_.get()))
        estimator = makePtr<detail::AffineBasedEstimator>();
    else
        estimator = makePtr<detail::HomographyBasedEstimator>();

    START_TIME(Estimate_Camera_Parameter_time);
    if (!(*estimator)(features_, pairwise_matches_, cameras_))
        return ERR_HOMOGRAPHY_EST_FAIL;
    STOP_TIME(Estimate_Camera_Parameter_time);

    for (size_t i = 0; i < cameras_.size(); ++i)
    {
        Mat R;
        cameras_[i].R.convertTo(R, CV_32F);
        cameras_[i].R = R;
        //LOGLN("Initial intrinsic parameters #" << indices_[i] + 1 << ":\n " << cameras_[i].K());
    }

    START_TIME(Bundle_Adjuster_time);
    bundle_adjuster_->setConfThresh(conf_thresh_);
    if (!(*bundle_adjuster_)(features_, pairwise_matches_, cameras_))
        return ERR_CAMERA_PARAMS_ADJUST_FAIL;
    STOP_TIME(Bundle_Adjuster_time);

    // Find median focal length and use it as final image scale
    std::vector<double> focals;
    for (size_t i = 0; i < cameras_.size(); ++i)
    {
        //LOGLN("Camera #" << indices_[i] + 1 << ":\n" << cameras_[i].K());
        focals.push_back(cameras_[i].focal);
    }

    std::sort(focals.begin(), focals.end());
    if (focals.size() % 2 == 1)
        warped_image_scale_ = static_cast<float>(focals[focals.size() / 2]);
    else
        warped_image_scale_ = static_cast<float>(focals[focals.size() / 2 - 1] + focals[focals.size() / 2]) * 0.5f;

    if (do_wave_correct_)
    {
        START_TIME(Wave_Correct_time);
        std::vector<Mat> rmats;
        for (size_t i = 0; i < cameras_.size(); ++i)
            rmats.push_back(cameras_[i].R.clone());
        detail::waveCorrect(rmats, wave_correct_kind_);
        for (size_t i = 0; i < cameras_.size(); ++i)
            cameras_[i].R = rmats[i];
        STOP_TIME(Wave_Correct_time);
    }

    return OK;
}


Ptr<Stitcher_mod> createStitcher(bool try_use_gpu)
{
    return Stitcher_mod::create(Stitcher_mod::PANORAMA, try_use_gpu);
}

Ptr<Stitcher_mod> createStitcherScans(bool try_use_gpu)
{
    return Stitcher_mod::create(Stitcher_mod::SCANS, try_use_gpu);
}
} // namespace cv
