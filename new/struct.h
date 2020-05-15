#pragma once
#include <opencv2/opencv.hpp>

enum video_display_mode
{
	//视频模式
	e_mode_video,

	//摄像头模式
	e_mode_camera
};

enum frame_handle_state
{
	//未处理
	e_un_handle,

	//检测中
	e_detec_handle,

	//处理完成
	e_finish_handle
};

struct frame_handle
{
	//视频帧
	cv::Mat frame;

	//处理状态
	enum frame_handle_state state;
};





