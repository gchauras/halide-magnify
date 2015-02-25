// HalideApps.cpp : Defines the entry point for the console application.
//

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

#pragma region Declarations

Var x("x"), y("y"), c("c"), w("w");

// Returns timing in milliseconds.
template<typename F0>
double timing(F0 f, int iterations = 1)
{
	auto t0 = currentTime();
	for (int i = 0; i < iterations; ++i)
		f();
	auto d = currentTime() - t0;
	return d / iterations;
}

// Prints and returns timing in milliseconds
template<typename F0>
double printTiming(F0 f, std::string message = "", int iterations = 1)
{
	if (!message.empty())
		std::cout << message << std::flush;
	double t = timing(f, iterations);
	std::cout << t << " ms" << std::endl;
	return t;
}

// Converts a Mat to an Image<uint8_t> (channels, width, height).
// Different order: channels = extent(0), width = extent(1), height = extent(2).
Image<uint8_t> toImage_uint8(const cv::Mat& mat)
{
	return Image<uint8_t>(Buffer(UInt(8), mat.channels(), mat.cols, mat.rows, 0, mat.data));;
}

// Converts a Mat to an Image<uint8_t> and reorders the data to be in the order (width, height, channels).
Image<uint8_t> toImage_uint8_reorder(const cv::Mat& mat)
{
	static Func convert;
	static ImageParam ip(UInt(8), 3);
	static Var xi, yi;

	if (!convert.defined())
	{
		convert(x, y, c) = ip(c, x, y);
		convert.vectorize(x, 4).parallel(y, 4);
	}

	ip.set(Buffer(UInt(8), mat.channels(), mat.cols, mat.rows, 0, mat.data));
	return convert.realize(mat.cols, mat.rows, mat.channels());
}

// Converts an Image<uint8_t> (channels, width, height) to a Mat.
cv::Mat toMat(Image<uint8_t> im)
{
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

#pragma endregion

int main_magnify(string filename="")
{
	RieszMagnifier magnifier(3, type_of<float>(), 5);
    magnifier.compileJIT(true);

	std::vector<double> filterA;
	std::vector<double> filterB;
	float alpha = 30.0f;
	double fps = 30.0;
	double freqCenter = 2;
	double freqWidth = .5;
	filter_util::computeFilter(fps, freqCenter, freqWidth, filterA, filterB);

    std::shared_ptr<VideoApp> app;
    if (!filename.empty()) {
        app.reset(new VideoApp(filename));
    } else {
        app.reset(new VideoApp);
    }

	std::vector<Image<float>> historyBuffer;
	for (int i = 0; i < magnifier.getPyramidLevels(); i++)
		historyBuffer.push_back(Image<float>(scaleSize(app->width(), i), scaleSize(app->height(), i), 7, 2));
	magnifier.bindJIT((float)filterA[1], (float)filterA[2], (float)filterB[0], (float)filterB[1], (float)filterB[2], alpha, historyBuffer);

	NamedWindow inputWindow("Input");
	NamedWindow resultWindow("Result");
	inputWindow.move(0, 0);
	resultWindow.move(10,10);
	Image<float> frame;
	Image<float> out(app->width(), app->height(), app->channels());
	double timeSum = 0;
	int frameCounter = -10;
	int pressedKey;
	for (int i = 0;; i++, frameCounter++)
	{
		frame = app->readFrame();
		if (frame.dimensions() == 0)
		{
			cv::waitKey();
			break;
		}

		double t = currentTime();
		// --- timing ---
		magnifier.process(frame, out);
		//std::cout << out(175, 226) << std::endl;
		// --- end timing ---
		double diff = currentTime() - t;
		//inputWindow.showImage(frame);
		resultWindow.showImage(out);
		std::cout << diff << " ms";

		if (frameCounter >= 0)
		{
			timeSum += diff / 1000.0;
			fps = (frameCounter + 1) / timeSum;
			std::cout << "\t(" << fps << " FPS)"
				<< "\t(" << 1000 / fps << " ms)" << std::endl;

			if (frameCounter % 10 == 0)
			{
				// Update fps
				filter_util::computeFilter(fps, freqCenter, freqWidth, filterA, filterB);
				magnifier.bindJIT((float)filterA[1], (float)filterA[2], (float)filterB[0], (float)filterB[1], (float)filterB[2], alpha, historyBuffer);
			}
		}
		else
		{
			std::cout << std::endl;
		}
		if ((pressedKey = cv::waitKey(30)) >= 0) {
			if (pressedKey == 45)	// minus
			{
				freqCenter = std::max(freqWidth, freqCenter - 0.5);
				std::cout << "Freq center is now " << freqCenter << std::endl;
				filter_util::computeFilter(fps, freqCenter, freqWidth, filterA, filterB);
				magnifier.bindJIT((float)filterA[1], (float)filterA[2], (float)filterB[0], (float)filterB[1], (float)filterB[2], alpha, historyBuffer);
			}
			else if (pressedKey == 43)	// plus
			{
				freqCenter += 0.5;
				std::cout << "Freq center is now " << freqCenter << std::endl;
				filter_util::computeFilter(fps, freqCenter, freqWidth, filterA, filterB);
				magnifier.bindJIT((float)filterA[1], (float)filterA[2], (float)filterB[0], (float)filterB[1], (float)filterB[2], alpha, historyBuffer);
			}
			else if (pressedKey == 97)	// a
			{
				// Increase alpha
				alpha -= 10;
				std::cout << "Alpha is now " << alpha << std::endl;
				magnifier.bindJIT((float)filterA[1], (float)filterA[2], (float)filterB[0], (float)filterB[1], (float)filterB[2], alpha, historyBuffer);
			}
            else if (pressedKey == 65)	// A
			{
				// Decrease alpha
				alpha += 10;
				std::cout << "Alpha is now " << alpha << std::endl;
				magnifier.bindJIT((float)filterA[1], (float)filterA[2], (float)filterB[0], (float)filterB[1], (float)filterB[2], alpha, historyBuffer);
			}
			else if (pressedKey == 27)
				break;
		}
	}

	return 0;
}

int main(int argc, char** argv) {
    if (argc>=2)	{
        return main_magnify(argv[1]);
	} else {
        return main_magnify();
    }
    return EXIT_SUCCESS;
}
