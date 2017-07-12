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
namespace
{

const int LSBPthreshold = 12;
const float LSBPtau = 0.03f;

inline float L2sqdist(const Point3f& a) {
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
        phi = float(CV_PI / 3);
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

inline void LSBPset(const Mat& localSVDValues, const Size& sz, uint32_t& descVal, float centerVal, int n, int i, int j) {
    if (i >= 0 && j >= 0 && i < sz.height && j < sz.width && std::abs(localSVDValues.at<float>(i, j) - centerVal) > LSBPtau)
        descVal |= uint32_t(1U) << n;
}

void removeNoise(Mat& fgMask, const Mat& compMask, const size_t threshold, const uint8_t filler) {
    const Size sz = fgMask.size();
    Mat labels;
    const int nComponents = connectedComponents(compMask, labels, 8, CV_32S);
    std::vector<size_t> compArea(nComponents, 0);

    for (int i = 0; i < sz.height; ++i)
        for (int j = 0; j < sz.width; ++j)
            ++compArea[labels.at<int>(i, j)];

    for (int i = 0; i < sz.height; ++i)
        for (int j = 0; j < sz.width; ++j)
            if (compArea[labels.at<int>(i, j)] < threshold)
                fgMask.at<uint8_t>(i, j) = filler;
}

class BackgroundSample {
public:
    Point3f color;
    uint32_t desc;
    uint64_t time;
    uint64_t hits;

    BackgroundSample(Point3f c = Point3f(), uint32_t d = 0, uint64_t t = 0, uint64_t h = 0) : color(c), desc(d), time(t), hits(h) {}
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

    float findClosest(int i, int j, const Point3f& color, int& indOut) const {
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

    Point3f getMean(int i, int j, uint64_t threshold) const {
        const int end = i * stride + (j + 1) * nSamples;
        Point3f acc(0, 0, 0);
        int cnt = 0;
        for (int k = i * stride + j * nSamples; k < end; ++k) {
            if (samples[k].hits > threshold) {
                acc += samples[k].color;
                ++cnt;
            }
        }
        if (cnt == 0) {
            cnt = nSamples;
            for (int k = i * stride + j * nSamples; k < end; ++k)
                acc += samples[k].color;
        }
        acc.x /= cnt;
        acc.y /= cnt;
        acc.z /= cnt;
        return acc;
    }
};

} // namespace

void BackgroundSubtractorLSBPDesc::calcLocalSVDValues(OutputArray _localSVDValues, const Mat& frame) {
    Mat frameGray;
    const Size sz = frame.size();
    _localSVDValues.create(sz, CV_32F);
    Mat localSVDValues = _localSVDValues.getMat();
    localSVDValues = 0.0f;

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

void BackgroundSubtractorLSBPDesc::computeFromLocalSVDValues(OutputArray _desc, const Mat& localSVDValues) {
    const Size sz = localSVDValues.size();
    _desc.create(sz, CV_32S);
    Mat desc = _desc.getMat();

    for (int i = 0; i < sz.height; ++i)
        for (int j = 0; j < sz.width; ++j) {
            uint32_t& descVal = desc.at<uint32_t>(i, j);
            descVal = 0;
            const float centerVal = localSVDValues.at<float>(i, j);
            int n = 0;

            for (int k = 1; k <= 4; ++k) {
                LSBPset(localSVDValues, sz, descVal, centerVal, n, i - k, j);
                ++n;
                LSBPset(localSVDValues, sz, descVal, centerVal, n, i + k, j);
                ++n;

                LSBPset(localSVDValues, sz, descVal, centerVal, n, i, j - k);
                ++n;
                LSBPset(localSVDValues, sz, descVal, centerVal, n, i, j + k);
                ++n;

                LSBPset(localSVDValues, sz, descVal, centerVal, n, i - k, j - k);
                ++n;
                LSBPset(localSVDValues, sz, descVal, centerVal, n, i + k, j + k);
                ++n;

                LSBPset(localSVDValues, sz, descVal, centerVal, n, i - k, j + k);
                ++n;
                LSBPset(localSVDValues, sz, descVal, centerVal, n, i + k, j - k);
                ++n;
            }
        }
}

void BackgroundSubtractorLSBPDesc::compute(OutputArray desc, const Mat& frame) {
    Mat localSVDValues;
    calcLocalSVDValues(localSVDValues, frame);
    compute(desc, localSVDValues);
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
    const float blinkingSupressionDecay;
    const float blinkingSupressionMultiplier;
    const float noiseRemovalThresholdFacBG;
    const float noiseRemovalThresholdFacFG;
    const bool useDescriptors;
    Mat distMovingAvg;
    Mat prevFgMask;
    Mat blinkingSupression;
    RNG rng;

    void postprocessing(Mat& fgMask);

public:
    BackgroundSubtractorLSBPImpl(int nSamples,
                                 float replaceRate,
                                 float propagationRate,
                                 uint64_t hitsThreshold,
                                 float alpha,
                                 float beta,
                                 float blinkingSupressionDecay,
                                 float blinkingSupressionMultiplier,
                                 float noiseRemovalThresholdFacBG,
                                 float noiseRemovalThresholdFacFG,
                                 bool useDescriptors);

    CV_WRAP virtual void apply(InputArray image, OutputArray fgmask, double learningRate = -1);

    CV_WRAP virtual void getBackgroundImage(OutputArray backgroundImage) const;
};

BackgroundSubtractorLSBPImpl::BackgroundSubtractorLSBPImpl(int _nSamples,
                                                           float _replaceRate,
                                                           float _propagationRate,
                                                           uint64_t _hitsThreshold,
                                                           float _alpha,
                                                           float _beta,
                                                           float _blinkingSupressionDecay,
                                                           float _blinkingSupressionMultiplier,
                                                           float _noiseRemovalThresholdFacBG,
                                                           float _noiseRemovalThresholdFacFG,
                                                           bool _useDescriptors)
: currentTime(0),
  nSamples(_nSamples),
  replaceRate(_replaceRate),
  propagationRate(_propagationRate),
  hitsThreshold(_hitsThreshold),
  alpha(_alpha),
  beta(_beta),
  blinkingSupressionDecay(_blinkingSupressionDecay),
  blinkingSupressionMultiplier(_blinkingSupressionMultiplier),
  noiseRemovalThresholdFacBG(_noiseRemovalThresholdFacBG),
  noiseRemovalThresholdFacFG(_noiseRemovalThresholdFacFG),
  useDescriptors(_useDescriptors) {
    CV_Assert(nSamples > 1 && nSamples < 1024);
    CV_Assert(replaceRate >= 0 && replaceRate <= 1);
    CV_Assert(propagationRate >= 0 && propagationRate <= 1);
    CV_Assert(blinkingSupressionDecay > 0 && blinkingSupressionDecay < 1);
    CV_Assert(noiseRemovalThresholdFacBG >= 0 && noiseRemovalThresholdFacBG < 0.5);
    CV_Assert(noiseRemovalThresholdFacFG >= 0 && noiseRemovalThresholdFacFG < 0.5);
}

void BackgroundSubtractorLSBPImpl::postprocessing(Mat& fgMask) {
    removeNoise(fgMask, fgMask, size_t(noiseRemovalThresholdFacBG * fgMask.size().area()), 0);
    Mat invFgMask = 255 - fgMask;
    removeNoise(fgMask, invFgMask, size_t(noiseRemovalThresholdFacFG * fgMask.size().area()), 255);

    GaussianBlur(fgMask, fgMask, Size(5, 5), 0);
    fgMask = fgMask > 127;
}

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

    Mat LSBPDesc(sz, CV_32S);
    LSBPDesc = 0;

    if (useDescriptors)
        BackgroundSubtractorLSBPDesc::compute(LSBPDesc, frame);

    CV_Assert(frame.channels() == 3);

    if (backgroundModel.empty()) {
        backgroundModel = makePtr<BackgroundModel>(sz, nSamples);
        distMovingAvg = Mat(sz, CV_32F, 0.005f);
        prevFgMask = Mat(sz, CV_8U);
        blinkingSupression = Mat(sz, CV_32F, 0.0f);

        for (int i = 0; i < sz.height; ++i)
            for (int j = 0; j < sz.width; ++j) {
                BackgroundSample sample(frame.at<Point3f>(i, j), LSBPDesc.at<uint32_t>(i, j));
                for (int k = 0; k < nSamples; ++k)
                    (* backgroundModel)(i, j, k) = sample;
            }
    }

    CV_Assert(backgroundModel->getSize() == sz);

    if (learningRate > 1 || learningRate < 0)
        learningRate = 0.1;

    for (int i = 0; i < sz.height; ++i)
        for (int j = 0; j < sz.width; ++j) {
            int k;
            const float minDist = backgroundModel->findClosest(i, j, frame.at<Point3f>(i, j), k);

            distMovingAvg.at<float>(i, j) *= 1 - float(learningRate);
            distMovingAvg.at<float>(i, j) += float(learningRate) * minDist;

            const float threshold = alpha * distMovingAvg.at<float>(i, j) + beta;
            BackgroundSample& sample = (* backgroundModel)(k);

            if (__builtin_popcount(sample.desc ^ LSBPDesc.at<uint32_t>(i, j)) > LSBPthreshold || minDist > threshold) {
                fgMask.at<uint8_t>(i, j) = 255;

                if (rng.uniform(0.0f, 1.0f) < replaceRate)
                    backgroundModel->replaceOldest(i, j, BackgroundSample(frame.at<Point3f>(i, j), LSBPDesc.at<uint32_t>(i, j), currentTime));
            }
            else {
                sample.color *= 1 - learningRate;
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

    cv::add(blinkingSupression, (fgMask != prevFgMask) / 255, blinkingSupression, cv::noArray(), CV_32F);
    blinkingSupression *= blinkingSupressionDecay;
    fgMask.copyTo(prevFgMask);
    Mat prob = blinkingSupression * (blinkingSupressionMultiplier * (1 - blinkingSupressionDecay) / blinkingSupressionDecay);

    for (int i = 0; i < sz.height; ++i)
        for (int j = 0; j < sz.width; ++j)
            if (rng.uniform(0.0f, 1.0f) < prob.at<float>(i, j))
                backgroundModel->replaceOldest(i, j, BackgroundSample(frame.at<Point3f>(i, j), LSBPDesc.at<uint32_t>(i, j), currentTime));

    this->postprocessing(fgMask);
}

void BackgroundSubtractorLSBPImpl::getBackgroundImage(OutputArray _backgroundImage) const {
    CV_Assert(!backgroundModel.empty());
    const Size sz = backgroundModel->getSize();
    _backgroundImage.create(sz, CV_8UC3);
    Mat backgroundImage = _backgroundImage.getMat();
    for (int i = 0; i < sz.height; ++i)
        for (int j = 0; j < sz.width; ++j)
            backgroundImage.at< Point3_<uint8_t> >(i, j) = backgroundModel->getMean(i, j, hitsThreshold) * 255;
}

Ptr<BackgroundSubtractorLSBP> createBackgroundSubtractorLSBP(int nSamples,
                                                             float replaceRate,
                                                             float propagationRate,
                                                             uint64_t hitsThreshold,
                                                             float alpha,
                                                             float beta,
                                                             float blinkingSupressionDecay,
                                                             float blinkingSupressionMultiplier,
                                                             float noiseRemovalThresholdFacBG,
                                                             float noiseRemovalThresholdFacFG,
                                                             bool useDescriptors) {
    return makePtr<BackgroundSubtractorLSBPImpl>(nSamples,
                                                 replaceRate,
                                                 propagationRate,
                                                 hitsThreshold,
                                                 alpha,
                                                 beta,
                                                 blinkingSupressionDecay,
                                                 blinkingSupressionMultiplier,
                                                 noiseRemovalThresholdFacBG,
                                                 noiseRemovalThresholdFacFG,
                                                 useDescriptors);
}

} // namespace bgsegm
} // namespace cv
