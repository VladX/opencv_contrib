#include "test_precomp.hpp"
#include <set>

using namespace std;
using namespace cv;
using namespace cvtest;
using namespace cv::bgsegm;

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
    BackgroundSubtractorLSBPDesc::calcLocalSVDValues(lsv2, input);

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
    cv::bgsegm::BackgroundSubtractorLSBPDesc::calcLocalSVDValues(lsv, input);

    EXPECT_LE(std::abs(lsv.at<float>(1, 1) - 0.0903614f), 0.001f);

    input = 1;
    cv::bgsegm::BackgroundSubtractorLSBPDesc::calcLocalSVDValues(lsv, input);

    EXPECT_LE(std::abs(lsv.at<float>(1, 1) - 0.0f), 0.001f);
}

TEST(BackgroundSubtractor_LSBP, Discrimination)
{
    Mat lena = imread(getLenaImagePath());
    Mat lsv;

    lena.convertTo(lena, CV_32FC3);

    cv::bgsegm::BackgroundSubtractorLSBPDesc::calcLocalSVDValues(lsv, lena);

    Scalar mean, var;
    meanStdDev(lsv, mean, var);

    EXPECT_GE(mean[0], 0.02);
    EXPECT_LE(mean[0], 0.04);
    EXPECT_GE(var[0], 0.03);

    Mat desc;
    cv::bgsegm::BackgroundSubtractorLSBPDesc::computeFromLocalSVDValues(desc, lsv);
    Size sz = desc.size();
    std::set<uint32_t> distinctive_elements;

    for (int i = 0; i < sz.height; ++i)
        for (int j = 0; j < sz.width; ++j)
            distinctive_elements.insert(desc.at<uint32_t>(i, j));

    EXPECT_GE(distinctive_elements.size(), 50000U);
}
