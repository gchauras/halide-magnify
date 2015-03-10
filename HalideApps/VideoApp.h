#pragma once

class VideoApp {
public:
	VideoApp(int scaleFactor=1, bool loop=false);
	VideoApp(std::string filename, bool loop);

	int width()     { return scaleFactor * cap.get(CV_CAP_PROP_FRAME_WIDTH); }
	int height()    { return scaleFactor * cap.get(CV_CAP_PROP_FRAME_HEIGHT);}
	int channels()  { return 3; }
	float fps()     { return cap.get(CV_CAP_PROP_FPS); }

	Halide::Image<float> readFrame(float position=-1.0f);
	Halide::Image<uint8_t> readFrame_uint8();

private:
	int scaleFactor;
    bool allow_looping;
	cv::VideoCapture cap;
};

class OutVideo {
public:
	OutVideo(std::string filename, float fps, int width, int height);
	void writeFrame(cv::Mat out);

private:
	cv::VideoWriter writer;
};
