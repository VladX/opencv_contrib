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
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

#include "precomp.hpp"
#include <iostream>

namespace cv
{
namespace bgsegm
{

inline float L2sqdist(const Vec4f& a) {
    return a.dot(a);
}

inline float det3x3(float a11, float a12, float a13, float a22, float a23, float a33) {
    return a11 * (a22 * a33 - a23 * a23) + a12 * (2 * a13 * a23 - a33 * a12) - a13 * a13 * a22;
}

inline float localSVD(float a11, float a12, float a13, float a21, float a22, float a23, float a31, float a32, float a33) {
    float b11 = a11 * a11 + a12 * a12 + a13 * a13;
    float b12 = a11 * a21 + a12 * a22 + a13 * a23;
    float b13 = a11 * a31 + a12 * a32 + a13 * a33;
    float b22 = a21 * a21 + a22 * a22 + a23 * a23;
    float b23 = a21 * a31 + a22 * a32 + a23 * a33;
    float b33 = a31 * a31 + a32 * a32 + a33 * a33;
    const float q = (b11 + b22 + b33) / 3;

    b11 -= q;
    b22 -= q;
    b33 -= q;

    float p = std::sqrt((b11 * b11 + b22 * b22 + b33 * b33 + 2 * (b12 * b12 + b13 * b13 + b23 * b23)) / 6);

    if (p == 0)
        return 0;

    const float pi = 1 / p;
    const float r = det3x3(pi * b11, pi * b12, pi * b13, pi * b22, pi * b23, pi * b33) / 2;
    float phi;

    if (r <= -1)
        phi = CV_PI / 3;
    else if (r >= 1)
        phi = 0;
    else
        phi = std::acos(r) / 3;

    p *= 2;
    const float e1 = q + p * std::cos(phi);
    float e2, e3;

    if (e1 < 3 * q) {
        e3 = std::max(q + p * std::cos(phi + float(2 * CV_PI / 3)), 0.0f);
        e2 = std::max(3 * q - e1 - e3, 0.0f);
    }
    else {
        e2 = 0;
        e3 = 0;
    }

    return std::sqrt(e2 / e1) + std::sqrt(e3 / e1);
}

inline void calcLocalSVDValues(Mat& localSVDValues, const Mat& frame) {
    Mat frameGray;
    const Size sz = frame.size();

    cvtColor(frame, frameGray, COLOR_BGR2GRAY);

    for (int i = 1; i < sz.height - 1; ++i)
        for (int j = 1; j < sz.width - 1; ++j) {
            localSVDValues.at<float>(i, j) = localSVD(
                frameGray.at<float>(i - 1, j - 1), frameGray.at<float>(i - 1, j), frameGray.at<float>(i - 1, j + 1),
                frameGray.at<float>(i, j - 1), frameGray.at<float>(i, j), frameGray.at<float>(i, j + 1),
                frameGray.at<float>(i + 1, j - 1), frameGray.at<float>(i + 1, j), frameGray.at<float>(i + 1, j + 1));
        }

    for (int i = 1; i < sz.height - 1; ++i) {
        localSVDValues.at<float>(i, 0) = localSVD(
            frameGray.at<float>(i - 1, 0), frameGray.at<float>(i - 1, 0), frameGray.at<float>(i - 1, 1),
            frameGray.at<float>(i, 0), frameGray.at<float>(i, 0), frameGray.at<float>(i, 1),
            frameGray.at<float>(i + 1, 0), frameGray.at<float>(i + 1, 0), frameGray.at<float>(i + 1, 1));

        localSVDValues.at<float>(i, sz.width - 1) = localSVD(
            frameGray.at<float>(i - 1, sz.width - 2), frameGray.at<float>(i - 1, sz.width - 1), frameGray.at<float>(i - 1, sz.width - 1),
            frameGray.at<float>(i, sz.width - 2), frameGray.at<float>(i, sz.width - 1), frameGray.at<float>(i, sz.width - 1),
            frameGray.at<float>(i + 1, sz.width - 2), frameGray.at<float>(i + 1, sz.width - 1), frameGray.at<float>(i + 1, sz.width - 1));
    }

    for (int j = 1; j < sz.width - 1; ++j) {
        localSVDValues.at<float>(0, j) = localSVD(
            frameGray.at<float>(0, j - 1), frameGray.at<float>(0, j), frameGray.at<float>(0, j + 1),
            frameGray.at<float>(0, j - 1), frameGray.at<float>(0, j), frameGray.at<float>(0, j + 1),
            frameGray.at<float>(1, j - 1), frameGray.at<float>(1, j), frameGray.at<float>(1, j + 1));
        localSVDValues.at<float>(sz.height - 1, j) = localSVD(
            frameGray.at<float>(sz.height - 2, j - 1), frameGray.at<float>(sz.height - 2, j), frameGray.at<float>(sz.height - 2, j + 1),
            frameGray.at<float>(sz.height - 1, j - 1), frameGray.at<float>(sz.height - 1, j), frameGray.at<float>(sz.height - 1, j + 1),
            frameGray.at<float>(sz.height - 1, j - 1), frameGray.at<float>(sz.height - 1, j), frameGray.at<float>(sz.height - 1, j + 1));
    }
}

class BackgroundSample {
public:
    Vec4f color;
    uint64_t time;
    uint64_t hits;

    BackgroundSample(Vec4f c = Vec4f(), uint64_t t = 0, uint64_t h = 0) : color(c), time(t), hits(h) {}
};

class BackgroundModel {
private:
    std::vector<BackgroundSample> samples;
    const Size size;
    const int nSamples;
    const int stride;

public:
    BackgroundModel(Size sz, int S) : size(sz), nSamples(S), stride(sz.width * S) {
        samples.resize(sz.area() * S);
    }

    const BackgroundSample& operator()(int k) const {
        return samples[k];
    }

    BackgroundSample& operator()(int k) {
        return samples[k];
    }

    const BackgroundSample& operator()(int i, int j, int k) const {
        return samples[i * stride + j * nSamples + k];
    }

    BackgroundSample& operator()(int i, int j, int k) {
        return samples[i * stride + j * nSamples + k];
    }

    Size getSize() const {
        return size;
    }

    float findClosest(int i, int j, const Vec4f& color, int& indOut) const {
        const int end = i * stride + (j + 1) * nSamples;
        int minInd = i * stride + j * nSamples;
        float minDist = L2sqdist(color - samples[minInd].color);
        for (int k = minInd + 1; k < end; ++k) {
            const float dist = L2sqdist(color - samples[k].color);
            if (dist < minDist) {
                minInd = k;
                minDist = dist;
            }
        }
        indOut = minInd;
        return minDist;
    }

    void replaceOldest(int i, int j, const BackgroundSample& sample) {
        const int end = i * stride + (j + 1) * nSamples;
        int minInd = i * stride + j * nSamples;
        for (int k = minInd + 1; k < end; ++k) {
            if (samples[k].time < samples[minInd].time)
                minInd = k;
        }
        samples[minInd] = sample;
    }
};

inline void addChannel(Mat& img, Mat& ch) {
    std::vector<Mat> matChannels;
    cv::split(img, matChannels);
    matChannels.push_back(ch);
    cv::merge(matChannels, img);
}

class BackgroundSubtractorLSBPImpl : public BackgroundSubtractorLSBP {
private:
    Ptr<BackgroundModel> backgroundModel;
    uint64_t currentTime;
    const int nSamples;
    const float replaceRate;
    const float propagationRate;
    const uint64_t hitsThreshold;
    const float alpha;
    const float beta;
    RNG rng;

public:
    BackgroundSubtractorLSBPImpl(int nSamples, float replaceRate, float propagationRate, uint64_t hitsThreshold, float alpha, float beta);

    CV_WRAP virtual void apply(InputArray image, OutputArray fgmask, double learningRate = -1);

    CV_WRAP virtual void getBackgroundImage(OutputArray backgroundImage) const;
};

BackgroundSubtractorLSBPImpl::BackgroundSubtractorLSBPImpl(int _nSamples, float _replaceRate, float _propagationRate, uint64_t _hitsThreshold, float _alpha, float _beta) : currentTime(0), nSamples(_nSamples), replaceRate(_replaceRate), propagationRate(_propagationRate), hitsThreshold(_hitsThreshold), alpha(_alpha), beta(_beta)
{}

void BackgroundSubtractorLSBPImpl::apply(InputArray _image, OutputArray _fgmask, double learningRate)
{
    const Size sz = _image.size();
    _fgmask.create(sz, CV_8U);
    Mat fgMask = _fgmask.getMat();

    Mat frame = _image.getMat();

    CV_Assert(frame.depth() == CV_8U || frame.depth() == CV_32F);
    CV_Assert(frame.channels() == 1 || frame.channels() == 3);

    if (frame.channels() != 3)
        cvtColor(frame, frame, COLOR_GRAY2BGR);

    if (frame.depth() != CV_32F) {
        frame.convertTo(frame, CV_32F);
        frame /= 255;
    }

    Mat localSVDValues(sz, CV_32F, 0.0f);
    //calcLocalSVDValues(localSVDValues, frame);
    //localSVDValues *= 2;
    addChannel(frame, localSVDValues);

    CV_Assert(frame.channels() == 4);

    if (backgroundModel.empty()) {
        backgroundModel = makePtr<BackgroundModel>(sz, nSamples);

        for (int i = 0; i < sz.height; ++i)
            for (int j = 0; j < sz.width; ++j) {
                BackgroundSample sample(frame.at<Vec4f>(i, j));
                for (int k = 0; k < nSamples; ++k)
                    (* backgroundModel)(i, j, k) = sample;
            }
    }

    CV_Assert(backgroundModel->getSize() == sz);

    if (learningRate > 1 || learningRate < 0)
        learningRate = 0.1;

    const float movingAvgLR = learningRate / 2;
    const float distEPS = 0.0001f;
    Mat distMovingAvg(sz, CV_32F, 0.005f);
    Mat distMovingVar(sz, CV_32F, 0.0f);

    for (int i = 0; i < sz.height; ++i)
        for (int j = 0; j < sz.width; ++j) {
            int k;
            const float minDist = backgroundModel->findClosest(i, j, frame.at<Vec4f>(i, j), k);

            distMovingAvg.at<float>(i, j) *= 1 - movingAvgLR;
            distMovingAvg.at<float>(i, j) += movingAvgLR * minDist;

            distMovingVar.at<float>(i, j) *= 1 - learningRate;
            distMovingVar.at<float>(i, j) += learningRate * std::abs(distMovingAvg.at<float>(i, j) - minDist);

            if (minDist > alpha * distMovingAvg.at<float>(i, j) + beta * distMovingVar.at<float>(i, j) + distEPS) {
                fgMask.at<uint8_t>(i, j) = 255;

                if (rng.uniform(0.0f, 1.0f) < replaceRate)
                    backgroundModel->replaceOldest(i, j, BackgroundSample(frame.at<Vec4f>(i, j), currentTime));
            }
            else {
                BackgroundSample& sample = (* backgroundModel)(k);
                sample.color *= 1 - learningRate;
                sample.color += learningRate * frame.at<Vec4f>(i, j);
                sample.time = currentTime;
                ++sample.hits;

                // Propagation to neighbors
                if (sample.hits > hitsThreshold && rng.uniform(0.0f, 1.0f) < propagationRate) {
                    if (i + 1 < sz.height)
                        backgroundModel->replaceOldest(i + 1, j, sample);
                    if (j + 1 < sz.width)
                        backgroundModel->replaceOldest(i, j + 1, sample);
                    if (i > 0)
                        backgroundModel->replaceOldest(i - 1, j, sample);
                    if (j > 0)
                        backgroundModel->replaceOldest(i, j - 1, sample);
                }

                fgMask.at<uint8_t>(i, j) = 0;
            }
        }

    ++currentTime;

    GaussianBlur(fgMask, fgMask, Size(5, 5), 0);
    fgMask = fgMask > 127;
}

void BackgroundSubtractorLSBPImpl::getBackgroundImage(OutputArray _backgroundImage) const
{
    _backgroundImage.create(backgroundModel->getSize(), CV_8U);
    Mat backgroundImage = _backgroundImage.getMat();
    backgroundImage = Scalar(0);
}

Ptr<BackgroundSubtractorLSBP> createBackgroundSubtractorLSBP(int nSamples, float replaceRate, float propagationRate, uint64_t hitsThreshold, float alpha, float beta)
{
    return makePtr<BackgroundSubtractorLSBPImpl>(nSamples, replaceRate, propagationRate, hitsThreshold, alpha, beta);
}

}
}
