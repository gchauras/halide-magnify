#include "stdafx.h"
#include "VideoApp.h"

using namespace Halide;

VideoApp::VideoApp(int scaleFactor, bool loop) : scaleFactor(scaleFactor), allow_looping(loop), cap(0) {
	if (!cap.isOpened()) {
		throw std::runtime_error("Cannot open webcam.");
    }
}

VideoApp::VideoApp(std::string filename, bool loop) : scaleFactor(1), allow_looping(loop), cap(filename) {
	if (!cap.isOpened()) {
		throw std::runtime_error("Cannot open file.");
    }
}

Image<float> VideoApp::readFrame(float position) {
	static Func convert("convertFromMat");
	static ImageParam ip(UInt(8), 3);
	static Var x("x"), y("y"), c("c");

	if (!convert.defined()) {
		convert(x,y,c) = ip(c, x/scaleFactor, y/scaleFactor) / 255.0f;
		convert.vectorize(x,4).parallel(y,4);
	}

	cv::Mat frame;

    if (position<0.0f || position>1.0f) {
        cap >> frame;
        if (frame.empty()) {
            if (allow_looping) {
                cap.set(CV_CAP_PROP_POS_FRAMES, 0);
                cap >> frame;
            }
            if (frame.empty()) {
                return Image<float>();
            }
        }
    } else {
        float num_frames = cap.get(CV_CAP_PROP_FRAME_COUNT);
        float curr_frame = cap.get(CV_CAP_PROP_POS_FRAMES);
        float seek_frame = floor(num_frames*position);
        cap.set(CV_CAP_PROP_POS_FRAMES, seek_frame);
        cap >> frame;
        cap.set(CV_CAP_PROP_POS_FRAMES, curr_frame);
    }

	ip.set(Buffer(UInt(8), frame.channels(), frame.cols, frame.rows, 0, frame.data));
	return convert.realize(scaleFactor*frame.cols, scaleFactor*frame.rows, frame.channels());
}

// -----------------------------------------------------------------------------

OutVideo::OutVideo(std::string filename, float fps, int width, int height) {
    writer.open(filename, CV_FOURCC('D','X','5','0'), fps,
            cv::Size(width,height), true);
    if (!writer.isOpened()) {
		throw std::runtime_error("Cannot open video file for writing.");
    }
}

void OutVideo::writeFrame(cv::Mat out) {
    if (!writer.isOpened()) {
		throw std::runtime_error("Cannot open video file for writing.");
    }
    writer << out;
}
