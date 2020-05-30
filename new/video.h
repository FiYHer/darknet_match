#pragma once

#include "image_opencv.h"
#include "opencv2/opencv.hpp"

#include "struct.h"
#include "detect.h"

#include <iostream>
#include <mutex>
#include <list>

#include <process.h>

class video
{
private:
	static video* m_static_this;

private:
	//正在读取视频帧
	bool m_reading;

	//正在检测视频帧
	bool m_detecting;

	//视频文件路径
	char* m_path;

	//摄像头索引
	int m_index;

	//播放模式
	enum video_display_mode m_mode;

	//视频播放
	cv::VideoCapture m_capture;

	//视频播放互斥
	std::mutex m_capture_mutex;

	//视频帧
	std::list<frame_handle*> m_frames;

	//视频帧互斥
	std::mutex m_frame_mutex;

	//暂停视频播放
	bool m_pause_video;

	//fps
	double m_display_fps;

	//物体检测模型
	object_detect m_detect_model;

	//区域列表
	std::vector<region_info> m_regions;

	//区域互斥
	std::mutex m_region_mutex;

private:
	//检测人流量
	struct calc_people_info m_calc_people;

	//检测车流量
	struct calc_car_info m_calc_car;

private:
	//读取视频帧线程
	static void __cdecl read_frame_thread(void* data);

	//检测视频帧线程
	static void __cdecl detect_frame_thread(void* data);

	//更新FPS
	void update_fps() noexcept;

	//读取一帧视频图像
	bool per_frame();

	//转化为实际坐标
	void box_to_pos(box b, int w, int h, int& left, int& top, int& right, int& bot);

	//判断两个矩阵的相交度
	bool is_coincide_rate(box a, box b, float value = 0.5f);

public:
	//判断是否读取视频帧中
	bool get_is_reading() const noexcept;

	//判断是否检测视频帧中
	bool get_is_detecting() const noexcept;

	//设置读取视频帧状态
	void set_reading(bool state) noexcept;

	//设置检测视频帧状态
	void set_detecting(bool state) noexcept;

	//获取视频文件路径
	const char* get_path() const noexcept;

	//获取摄像头索引
	int get_index() const noexcept;

	//获取当前模式
	video_display_mode get_mode() const noexcept;

	//获取视频类指针
	cv::VideoCapture* get_capture() noexcept;

	//获取视频帧列表指针
	std::list<frame_handle*>* get_frames() noexcept;

	//视频类互斥
	void entry_capture_mutex() noexcept;
	void leave_capture_mutex() noexcept;

	//视频帧互斥
	void entry_frame_mutex() noexcept;
	void leave_frame_mutex() noexcept;

	//设置暂停状态
	void set_payse_state() noexcept;

	//获取暂停状态
	bool get_pause_state() const noexcept;

	//获取FPS
	double get_display_fps() const noexcept;

	//获取检测类指针
	object_detect* get_detect_model() noexcept;

	//视频帧转化图像
	image to_image(cv::Mat frame, int out_w, int out_h, int out_c) noexcept;

	//绘制方框和字体
	void draw_box_and_font(detection* detect, int count, cv::Mat* frame) noexcept;

	//获取区域列表
	std::vector<region_info> get_region_list() const noexcept;

	//区域互斥
	void entry_region_mutex() noexcept;
	void leave_region_mutex() noexcept;

	//加入区域到尾部
	void push_region_back(struct region_info& region) noexcept;

	//从尾部删除区域
	void pop_region_back() noexcept;

	//场景管理
	void scene_manager(detection* detect, int count, int w, int h) noexcept;

	//人流量统计场景
	void scene_calc_people(std::vector<box> b) noexcept;

	//车流量统计场景
	void scene_calc_car(std::vector<box> b) noexcept;

	//获取人流量结构
	struct calc_people_info* get_people_info_point() noexcept;

	//获取车流量结构
	struct calc_car_info* get_car_info_point() noexcept;

public:
	//设置视频路径
	bool set_video_path(const char* path) noexcept;

	//设置摄像头索引
	bool set_video_index(int index) noexcept;

	//读取视频帧
	struct frame_handle* get_video_frame() noexcept;

	//设置视频帧
	void set_frame_index(int index) noexcept;
	void set_frame_index(float rate) noexcept;

	//获取视频播放完成比率
	float get_finish_rate() noexcept;

	//获取一帧视频
	void get_per_video_frame(const char* path);
	void get_per_video_frame(int index);

public:
	video();
	~video();

	//开始播放视频
	bool start() noexcept;

	//暂停播放视频
	void pause() noexcept;

	//重新开始播放视频
	void restart() noexcept;

	//关闭视频播放
	void close() noexcept;
};
