#pragma once
#include <opencv2/opencv.hpp>

//对象类别索引
enum object_classes
{
	object_license_plate,//车牌
	object_car,//车子
	object_bus,//公交车
	objetc_person,//行人
	object_traffic_light//红绿灯
};

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

//矩阵区域信息
struct rect_info
{
	//
	int left, top, right, down;

	rect_info() : left(0), top(0), right(0), down(0) {}
	rect_info(int l, int t, int r, int d) : left(l), top(t), right(r), down(d) {}
	rect_info operator=(rect_info& t)
	{
		left = t.left;
		top = t.top;
		right = t.right;
		down = t.down;
		return *this;
	}
};

//区域信息
/// <summary>
/// 区域信息
/// </summary>
struct region_info
{
	//矩阵区域
	struct rect_info rect;

	//区域类型
	enum region_type type;

	//区域颜色
	unsigned int color;

	region_info() : type(region_bus), color(0) {}
	region_info(struct rect_info r, region_type t, unsigned int c)
	{
		rect = r;
		type = t;
		color = c;
	}
};

//检测人流量信息
struct calc_people_info
{
	//启用
	bool enable;

	//当前人流量
	unsigned int current_val;

	//最大人流量
	unsigned int max_val;

	calc_people_info() : enable(false) {}
};

struct calc_car_info
{
	//启用
	bool enable;

	calc_car_info() : enable(false) {}
};
