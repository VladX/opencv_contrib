#include "test_precomp.hpp"
#include <set>

using namespace std;
using namespace cv;
using namespace cvtest;

static string getDataDir() { return TS::ptr()->get_data_path(); }

static string getLenaImagePath() { return getDataDir() + "shared/lena.png"; }

// Simple synthetic illumination invariance test
TEST(BackgroundSubtractor_LSBP, IlluminationInvariance)
{
    RNG rng;
    Mat input(100, 100, CV_32FC3);

    rng.fill(input, RNG::UNIFORM, 0.0f, 0.1f);

    Mat lsv1, lsv2;
    cv::bgsegm::BackgroundSubtractorLSBPDesc::calcLocalSVDValues(lsv1, input);
    input *= 10;
    cv::bgsegm::BackgroundSubtractorLSBPDesc::calcLocalSVDValues(lsv2, input);

    ASSERT_LE(cv::norm(lsv1, lsv2), 0.04f);
}

TEST(BackgroundSubtractor_LSBP, Correctness)
{
    RNG rng;
    Mat input(3, 3, CV_32FC3);

    int n = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            input.at<Point3f>(i, j) = Point3f(n, n, n);
            ++n;
        }

    Mat lsv;
    bgsegm::BackgroundSubtractorLSBPDesc::calcLocalSVDValues(lsv, input);

    EXPECT_LE(std::abs(lsv.at<float>(1, 1) - 0.0903614f), 0.001f);

    input = 1;
    bgsegm::BackgroundSubtractorLSBPDesc::calcLocalSVDValues(lsv, input);

    EXPECT_LE(std::abs(lsv.at<float>(1, 1) - 0.0f), 0.001f);
}

TEST(BackgroundSubtractor_LSBP, Discrimination)
{
    Mat lena = imread(getLenaImagePath());
    Mat lsv;

    lena.convertTo(lena, CV_32FC3);

    bgsegm::BackgroundSubtractorLSBPDesc::calcLocalSVDValues(lsv, lena);

    Scalar mean, var;
    meanStdDev(lsv, mean, var);

    EXPECT_GE(mean[0], 0.02);
    EXPECT_LE(mean[0], 0.04);
    EXPECT_GE(var[0], 0.03);

    Mat desc;
    bgsegm::BackgroundSubtractorLSBPDesc::computeFromLocalSVDValues(desc, lsv);
    Size sz = desc.size();
    std::set<uint32_t> distinctive_elements;

    for (int i = 0; i < sz.height; ++i)
        for (int j = 0; j < sz.width; ++j)
            distinctive_elements.insert(desc.at<uint32_t>(i, j));

    EXPECT_GE(distinctive_elements.size(), 50000U);
}

static double scoreBitwiseReduce(const Mat& mask, const Mat& gtMask, uint8_t v1, uint8_t v2) {
    Mat result;
    cv::bitwise_and(mask == v1, gtMask == v2, result);
    return cv::sum(result)[0];
}

TEST(BackgroundSubtractor_LSBP, Accuracy)
{
    Ptr<bgsegm::BackgroundSubtractorLSBP> bgs = bgsegm::createBackgroundSubtractorLSBP();

    double f1_mean = 0;
    unsigned total = 0;

    for (int frameNum = 1; frameNum <= 900; ++frameNum) {
        char frameName[256], gtMaskName[256];
        sprintf(frameName, "bgsegm/highway/input/in%06d.jpg", frameNum);
        sprintf(gtMaskName, "bgsegm/highway/groundtruth/gt%06d.png", frameNum);

        Mat frame = imread(getDataDir() + frameName);
        Mat gtMask = imread(getDataDir() + gtMaskName, IMREAD_GRAYSCALE);

        Mat mask;
        bgs->apply(frame, mask);

        Size sz = frame.size();
        EXPECT_EQ(sz, gtMask.size());
        EXPECT_EQ(mask.type(), gtMask.type());
        EXPECT_EQ(mask.type(), CV_8U);

        const double tp = scoreBitwiseReduce(mask, gtMask, 255, 255);
        const double fp = scoreBitwiseReduce(mask, gtMask, 255, 0);
        const double fn = scoreBitwiseReduce(mask, gtMask, 0, 255);

        if (tp + fn + fp > 0) {
            const double f1_score = 2.0 * tp / (2.0 * tp + fn + fp);
            f1_mean += f1_score;
            ++total;
        }
    }

    f1_mean /= total;

    EXPECT_GE(f1_mean, 0.9);
}
