#include "stdafx.h"
#include "Util.h"
#include "VideoApp.h"
#include "NamedWindow.h"
#include "EulerianMagnifier.h"
#include "RieszMagnifier.h"
#include "filter_util.h"

using namespace Halide;
using std::vector;
using std::string;
using std::cerr;
using std::endl;

Var x("x"), y("y"), c("c"), w("w");

// Returns timing in milliseconds.
template<typename F0>
double timing(F0 f, int iterations = 1)
{
	auto t0 = currentTime();
	for (int i = 0; i < iterations; ++i) {
		f();
    }
	auto d = currentTime() - t0;
	return d / iterations;
}

// Prints and returns timing in milliseconds
template<typename F0>
double printTiming(F0 f, std::string message = "", int iterations = 1)
{
	if (!message.empty()) {
		cerr << message << std::flush;
    }
	double t = timing(f, iterations);
	cerr << t << " ms" << endl;
	return t;
}

// Converts a Mat to an Image<uint8_t> (channels, width, height).
// Different order: channels = extent(0), width = extent(1), height = extent(2).
Image<uint8_t> toImage_uint8(const cv::Mat& mat) {
	return Image<uint8_t>(Buffer(UInt(8), mat.channels(), mat.cols, mat.rows, 0, mat.data));;
}

// Converts a Mat to an Image<uint8_t> and reorders the data to be in the order (width, height, channels).
Image<uint8_t> toImage_uint8_reorder(const cv::Mat& mat) {
	static Func convert;
	static ImageParam ip(UInt(8), 3);
	static Var xi, yi;

	if (!convert.defined()) {
		convert(x, y, c) = ip(c, x, y);
		convert.vectorize(x, 4).parallel(y, 4);
	}

	ip.set(Buffer(UInt(8), mat.channels(), mat.cols, mat.rows, 0, mat.data));
	return convert.realize(mat.cols, mat.rows, mat.channels());
}

// Converts an Image<uint8_t> (channels, width, height) to a Mat.
cv::Mat toMat(Image<uint8_t> im) {
	return cv::Mat(im.extent(2), im.extent(1), CV_8UC3, im.data());
}

// Converts a Mat to an Image<float> and reorders the data to be in the order (width, height, channels).
Image<float> toImage_reorder(const cv::Mat& mat)
{
	static Func convert;
	static ImageParam ip(UInt(8), 3);
	static Var xi, yi;

	if (!convert.defined())
	{
		convert(x, y, c) = ip(c, x, y) / 255.0f;
		convert.vectorize(x, 4).parallel(y, 4);
	}

	ip.set(Buffer(UInt(8), mat.channels(), mat.cols, mat.rows, 0, mat.data));
	return convert.realize(mat.cols, mat.rows, mat.channels());
}

// Converts a reordered Image<uint8_t> (width, height) to a Mat (CV_8U).
cv::Mat toMat2d(Image<float> im)
{
	static Func convert;
	static ImageParam ip(Float(32), 2);
	static Var xi, yi;

	if (!convert.defined())
	{
		convert(x, y) = cast<uint8_t>(ip(x, y) * 255);
		convert.vectorize(x, 4).parallel(y, 4);
	}

	ip.set(im);
	cv::Mat mat(im.height(), im.width(), CV_8U, cv::Scalar(0));
	convert.realize(Buffer(UInt(8), im.width(), im.height(), 0, 0, mat.data));
	return mat;
}

// Converts a reordered Image<uint8_t> (width, height, channels) to a Mat (CV_8UC3).
cv::Mat toMat_reordered(Image<float> im)
{
	static Func convert;
	static ImageParam ip(Float(32), 3);
	static Var xi, yi;

	if (!convert.defined())
	{
		convert(c, x, y) = cast<uint8_t>(ip(x, y, c) * 255);
		convert.vectorize(x, 4).parallel(y, 4);
	}

	ip.set(im);
	cv::Mat mat(im.height(), im.width(), CV_8UC3, cv::Scalar(0));
	convert.realize(Buffer(UInt(8), im.channels(), im.width(), im.height(), 0, mat.data));
	return mat;
}


int main(int argc, char** argv) {
    bool   use_gui;
    string filename;
    if (argc == 3) {
        filename = argv[1];
        use_gui  = (atoi(argv[2]) == 0);
    } else {
        cerr << "Usage: phase_magnifier [input video] [0|1 0 for using GUI, 1 to dump result in a video]" << endl;
        return EXIT_FAILURE;
    }

	RieszMagnifier magnifier(3, type_of<float>(), 7);
    magnifier.compileJIT(true);

	std::vector<double> filterA;
	std::vector<double> filterB;
	float alpha = 0.0f;
	double fps = 30.0;
	double freqCenter = 2;
	double freqWidth = .5;
	filter_util::computeFilter(fps, freqCenter, freqWidth, filterA, filterB);

    std::shared_ptr<VideoApp> app;
    std::shared_ptr<OutVideo> result;
    std::shared_ptr<NamedWindow> resultWindow;

    if (!filename.empty()) {
        app.reset(new VideoApp(filename, use_gui));
    } else {
        app.reset(new VideoApp);
    }

    // show output or write to file
    if (use_gui) {
        resultWindow.reset(new NamedWindow("Result"));
    } else {
        result.reset(new OutVideo("out.avi", app->fps(), app->width(), app->height()));
    }

	std::vector<Image<float>> historyBuffer;
	for (int i=0; i<magnifier.getPyramidLevels(); i++) {
		historyBuffer.push_back(Image<float>(
                    scaleSize(app->width(), i),
                    scaleSize(app->height(), i), 7, 2));
    }

    magnifier.bindJIT(filterA[1], filterA[2],
            filterB[0], filterB[1], filterB[2], alpha, historyBuffer);

    // select a reference amplitude frame
    magnifier.compute_reference_amplitude(app->readFrame(0.5f));

	Image<float> frame;
	Image<float> out(app->width(), app->height(), app->channels());

    bool paused = false;
    double timeSum = 0;
    int frameCounter = -10;
	int pressedKey;

	for (int i=0;; i++, frameCounter++) {
        if (!paused) {
            frame = app->readFrame();
            if (frame.dimensions() == 0) {
                return EXIT_SUCCESS;
            }

            double t = currentTime();
            magnifier.process(frame, out);
            double diff = currentTime() - t;

            if (use_gui) {
                resultWindow->move(10,10);
                resultWindow->showImage(out);
            } else {
                result->writeFrame(toMat_reordered(out));
            }
            cerr << diff << " ms";

            if (frameCounter >= 0) {
                timeSum += diff / 1000.0;
                fps = (frameCounter + 1) / timeSum;
                cerr << "\t(" << fps << " FPS)" << "\t(" << 1000 / fps << " ms)" << endl;

                // Update fps
                if (frameCounter % 10 == 0) {
                    filter_util::computeFilter(fps, freqCenter, freqWidth, filterA, filterB);
                    magnifier.bindJIT((float)filterA[1], (float)filterA[2], (float)filterB[0], (float)filterB[1], (float)filterB[2], alpha, historyBuffer);
                }
            }
            else {
                cerr << endl;
            }
        }

        if ((pressedKey = cv::waitKey(30)) >= 0) {
            if (pressedKey == 45)	// minus
            {
                freqCenter = std::max(freqWidth, freqCenter - 0.5);
                cerr << "Freq center is now " << freqCenter << endl;
                filter_util::computeFilter(fps, freqCenter, freqWidth, filterA, filterB);
                magnifier.bindJIT((float)filterA[1], (float)filterA[2], (float)filterB[0], (float)filterB[1], (float)filterB[2], alpha, historyBuffer);
            }
            else if (pressedKey == 43)	// plus
            {
                freqCenter += 0.5;
                cerr << "Freq center is now " << freqCenter << endl;
                filter_util::computeFilter(fps, freqCenter, freqWidth, filterA, filterB);
                magnifier.bindJIT((float)filterA[1], (float)filterA[2], (float)filterB[0], (float)filterB[1], (float)filterB[2], alpha, historyBuffer);
            }
            else if (pressedKey == 97)	// a: decrease alpha
            {
                alpha += 10;
                cerr << "Alpha is now " << alpha << endl;
                magnifier.bindJIT((float)filterA[1], (float)filterA[2], (float)filterB[0], (float)filterB[1], (float)filterB[2], alpha, historyBuffer);
            }
            else if (pressedKey == 65)	// A: increase alpha
            {
                alpha += 10;
                cerr << "Alpha is now " << alpha << endl;
                magnifier.bindJIT((float)filterA[1], (float)filterA[2], (float)filterB[0], (float)filterB[1], (float)filterB[2], alpha, historyBuffer);
            }
            else if (pressedKey == 32) {
                paused = !paused;
            }
            else if (pressedKey == 27) {
                break;
            }
        }
    }

    return EXIT_SUCCESS;
}
