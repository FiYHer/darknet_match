#pragma once
#include <opencv2/opencv.hpp>

//视频播放类型
enum video_display_mode
{
	//视频模式
	e_mode_video,

	//摄像头模式
	e_mode_camera
};

//视频帧处理状态
enum frame_handle_state
{
	//未处理
	e_un_handle,

	//检测中
	e_detec_handle,

	//处理完成
	e_finish_handle
};

//视频帧处理信息
struct frame_handle
{
	//视频帧
	cv::Mat frame;

	//处理状态
	enum frame_handle_state state;
};

//区域类型
enum region_type
{
	//公交车道
	region_bus,

	//斑马线
	region_zebra_cross
};

struct region_info
{
	//
	int x, y, w, h;

	//区域类型
	enum region_type type;


};














