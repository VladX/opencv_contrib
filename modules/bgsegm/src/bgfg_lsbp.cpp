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
//                       (3-clause BSD License)
//                     For BackgroundSubtractorCNT
//               (Background Subtraction based on Counting)
//
// Copyright (C) 2016, Sagi Zeevi (www.theimpossiblecode.com), all rights reserved.
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

inline float L2sqdist(Point3f a, Point3f b) {
    a -= b;
    return a.dot(a);
}

class BackgroundSample {
public:
    Point3f color;
    uint64_t time;
    uint64_t hits;

    BackgroundSample(Point3f c = Point3f(), uint64_t t = 0, uint64_t h = 0) : color(c), time(t), hits(h) {}
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

    int findClosest(int i, int j, Point3f color, float threshold) const {
        const int end = i * stride + (j + 1) * nSamples;
        int minInd = i * stride + j * nSamples;
        float minDist = L2sqdist(color, samples[minInd].color);
        for (int k = minInd + 1; k < end; ++k) {
            const float dist = L2sqdist(color, samples[k].color);
            if (dist < minDist) {
                minInd = k;
                minDist = dist;
            }
        }
        return minDist < threshold ? minInd : -1;
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

class BackgroundSubtractorLSBPImpl : public BackgroundSubtractorLSBP {
private:
    Ptr<BackgroundModel> backgroundModel;
    uint64_t currentTime;
    const int nSamples;
    const float replaceRate;
    const float propagationRate;
    const uint64_t hitsThreshold;
    RNG rng;

public:
    BackgroundSubtractorLSBPImpl(int nSamples = 20, float replaceRate = 0.0075f, float propagationRate = 0.03f, uint64_t hitsThreshold = 32);

    CV_WRAP virtual void apply(InputArray image, OutputArray fgmask, double learningRate = -1);

    CV_WRAP virtual void getBackgroundImage(OutputArray backgroundImage) const;
};

BackgroundSubtractorLSBPImpl::BackgroundSubtractorLSBPImpl(int _nSamples, float _replaceRate, float _propagationRate, uint64_t _hitsThreshold) : currentTime(0), nSamples(_nSamples), replaceRate(_replaceRate), propagationRate(_propagationRate), hitsThreshold(_hitsThreshold)
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

    if (backgroundModel.empty()) {
        backgroundModel = makePtr<BackgroundModel>(sz, nSamples);

        for (int i = 0; i < sz.height; ++i)
            for (int j = 0; j < sz.width; ++j) {
                BackgroundSample sample(frame.at<Point3f>(i, j));
                for (int k = 0; k < nSamples; ++k)
                    (* backgroundModel)(i, j, k) = sample;
            }
    }

    CV_Assert(backgroundModel->getSize() == sz);

    if (learningRate > 1 || learningRate < 0)
        learningRate = 0.1;

    for (int i = 0; i < sz.height; ++i)
        for (int j = 0; j < sz.width; ++j) {
            const int k = backgroundModel->findClosest(i, j, frame.at<Point3f>(i, j), 0.01f);
            if (k == -1) {
                fgMask.at<uint8_t>(i, j) = 255;

                if (rng.uniform(0.0f, 1.0f) < replaceRate) {
                    backgroundModel->replaceOldest(i, j, BackgroundSample(frame.at<Point3f>(i, j), currentTime));
                }
            }
            else {
                BackgroundSample& sample = (* backgroundModel)(k);
                sample.color *= 1.0 - learningRate;
                sample.color += learningRate * frame.at<Point3f>(i, j);
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
}

void BackgroundSubtractorLSBPImpl::getBackgroundImage(OutputArray _backgroundImage) const
{
    _backgroundImage.create(backgroundModel->getSize(), CV_8U);
    Mat backgroundImage = _backgroundImage.getMat();
    backgroundImage = Scalar(0);
}

Ptr<BackgroundSubtractorLSBP> createBackgroundSubtractorLSBP()
{
    return makePtr<BackgroundSubtractorLSBPImpl>();
}

}
}
