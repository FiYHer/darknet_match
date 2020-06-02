#pragma once

#include "darknet.h"

#include <opencv2/opencv.hpp>
#include "help.h"

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
	//窗口宽高
	float w, h;

	//
	float left, top, right, down;

	rect_info() : left(0), top(0), right(0), down(0) {}
	rect_info(float l, float t, float r, float d) : left(l), top(t), right(r), down(d) {}
	rect_info operator=(rect_info& t)
	{
		left = t.left;
		top = t.top;
		right = t.right;
		down = t.down;
		return *this;
	}
};

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

	//返回规范化的位置信息
	box to_box_data() noexcept
	{
		box b;
		b.x = (rect.left - 8.0f) / (rect.w - 8.0f);
		b.y = (rect.top - 35.0f) / (rect.h - 35.0f);
		b.w = (rect.right - 8.0f) / (rect.w - 8.0f);
		b.h = (rect.down - 35.0f) / (rect.h - 35.0f);
		return b;
	}

	region_info() : type(region_bus), color(0) {}
	region_info(struct rect_info r, region_type t, unsigned int c)
	{
		rect = r;
		type = t;
		color = c;
	}
};

//检测人/车流量信息
struct calc_statistics_info
{
	//启用
	bool enable;

	//当前人/车流量
	unsigned int current_val;

	//最大人/车流量
	unsigned int max_val;

	//记录每分钟的人/车流量
	std::vector<unsigned int> val_list;

	void update_current_val(unsigned int val) { current_val = val; if (val > max_val) max_val = val; }

	void update_new_val(unsigned int val)
	{
		static int time_minute = get_current_minute();
		int current_minute = get_current_minute();

		if (time_minute != current_minute)
		{
			time_minute = current_minute;
			val_list.push_back(val);
		}
		else
		{
			int index = val_list.size() - 1;
			if (index == -1) val_list.push_back(val);
			else val_list[index] += val;
		}
	}

	void clear()
	{
		current_val = max_val = 0;
		val_list.clear();
	}

	calc_statistics_info() : enable(false) { clear(); }
};
