#include "test_precomp.hpp"

using namespace std;
using namespace cv;
using namespace cvtest;
using namespace cv::bgsegm;

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
