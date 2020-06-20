#pragma once

#include "image_opencv.h"
#include "opencv2/opencv.hpp"

#include "struct.h"
#include "detect.h"
#include "recognition.h"

#include <iostream>
#include <mutex>
#include <list>

#include <process.h>

/// <summary>
/// 视频播放类
/// </summary>
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

	//车牌识别模型
	object_recognition m_recognition_model;

	//区域列表
	std::vector<region_info> m_regions;

	//区域互斥
	std::mutex m_region_mutex;

private:
	//检测人流量
	struct calc_statistics_info m_calc_people;

	//检测车流量
	struct calc_statistics_info m_calc_car;

	//检测和识别车牌
	bool m_recognition_car_id;

private:
	/// <summary>
	/// 读取视频帧线程.
	/// </summary>
	/// <param name="data">无用.</param>
	/// <returns></returns>
	static void __cdecl read_frame_thread(void* data);

	/// <summary>
	/// 检测视频帧线程.
	/// </summary>
	/// <param name="data">无用.</param>
	/// <returns></returns>
	static void __cdecl detect_frame_thread(void* data);

	/// <summary>
	/// 更新FPS.
	/// </summary>
	/// <returns></returns>
	void update_fps() noexcept;

	/// <summary>
	/// 读取一帧视频图像.
	/// </summary>
	/// <returns>读取成功返回true，否则返回false</returns>
	bool per_frame();

	/// <summary>
	/// 转化为实际坐标.
	/// </summary>
	/// <param name="b">位置比率结构.</param>
	/// <param name="w">图像宽度.</param>
	/// <param name="h">图像高度.</param>
	/// <param name="left">返回最左点.</param>
	/// <param name="top">返回最上点.</param>
	/// <param name="right">返回最右点.</param>
	/// <param name="bot">返回最下点.</param>
	void box_to_pos(box b, int w, int h, int& left, int& top, int& right, int& bot);

	/// <summary>
	/// 判断两个矩阵的相交度.
	/// </summary>
	/// <param name="a">A位置比率结构.</param>
	/// <param name="b">B位置比率结构.</param>
	/// <param name="value">相交比率.</param>
	/// <returns>
	///   <c>true</c> a和b的相交比率大于value
	///   <c>false</c> a和b的相交比率小于value
	/// </returns>
	bool is_coincide_rate(box a, box b, float value = 0.5f);

public:
	/// <summary>
	/// 判断是否读取视频帧中.
	/// </summary>
	/// <returns>视频读取中返回true，否则返回false</returns>
	bool get_is_reading() const noexcept;

	/// <summary>
	/// 判断是否检测视频帧中.
	/// </summary>
	/// <returns>检测视频中返回true，否则返回false</returns>
	bool get_is_detecting() const noexcept;

	/// <summary>
	/// 设置读取视频帧状态.
	/// </summary>
	/// <param name="state">设置的状态.</param>
	/// <returns></returns>
	void set_reading(bool state) noexcept;

	/// <summary>
	/// 设置检测视频帧状态.
	/// </summary>
	/// <param name="state">设置的状态.</param>
	/// <returns></returns>
	void set_detecting(bool state) noexcept;

	/// <summary>
	/// 获取视频文件路径.
	/// </summary>
	/// <returns>返回视频文件字符串</returns>
	const char* get_path() const noexcept;

	/// <summary>
	/// 获取摄像头索引.
	/// </summary>
	/// <returns>返回摄像头索引</returns>
	int get_index() const noexcept;

	/// <summary>
	/// 获取当前模式.
	/// </summary>
	/// <returns>返回当前模型</returns>
	video_display_mode get_mode() const noexcept;

	/// <summary>
	/// 获取视频类指针.
	/// </summary>
	/// <returns>返回视频读取类指针</returns>
	cv::VideoCapture* get_capture() noexcept;

	/// <summary>
	/// 获取视频帧列表指针.
	/// </summary>
	/// <returns>返回视频帧列表的指针</returns>
	std::list<frame_handle*>* get_frames() noexcept;

	/// <summary>
	/// 视频类互斥.
	/// </summary>
	/// <returns></returns>
	void entry_capture_mutex() noexcept;
	void leave_capture_mutex() noexcept;

	/// <summary>
	/// 视频帧互斥.
	/// </summary>
	/// <returns></returns>
	void entry_frame_mutex() noexcept;
	void leave_frame_mutex() noexcept;

	/// <summary>
	/// 设置暂停状态.
	/// </summary>
	/// <returns></returns>
	void set_payse_state() noexcept;

	/// <summary>
	/// 获取暂停状态.
	/// </summary>
	/// <returns>返回暂停状态</returns>
	bool get_pause_state() const noexcept;

	/// <summary>
	/// 获取FPS.
	/// </summary>
	/// <returns>返回当前的fps</returns>
	double get_display_fps() const noexcept;

	/// <summary>
	/// 获取检测类指针.
	/// </summary>
	/// <returns>返回物体检测模型指针</returns>
	object_detect* get_detect_model() noexcept;

	/// <summary>
	/// 获取识别类指针.
	/// </summary>
	/// <returns>返回车牌识别模型指针</returns>
	object_recognition* get_recognition_model() noexcept;

	/// <summary>
	/// 视频帧cv::Mat转化image图像.
	/// </summary>
	/// <param name="frame">视频帧.</param>
	/// <param name="out_w">宽度.</param>
	/// <param name="out_h">高度.</param>
	/// <param name="out_c">通道数.</param>
	/// <returns>返回转化后的image</returns>
	image to_image(cv::Mat frame, int out_w, int out_h, int out_c) noexcept;

	/// <summary>
	/// 绘制方框和字体.
	/// </summary>
	/// <param name="detect">The detect.</param>
	/// <param name="count">The count.</param>
	/// <param name="frame">The frame.</param>
	/// <returns></returns>
	void draw_box_and_font(detection* detect, int count, cv::Mat* frame) noexcept;

	/// <summary>
	/// 绘制设置好的相关区域.
	/// </summary>
	/// <param name="frame">视频帧.</param>
	/// <returns></returns>
	void draw_regions(cv::Mat* frame) noexcept;

	/// <summary>
	/// 获取区域列表.
	/// </summary>
	/// <returns>返回区域结构列表</returns>
	std::vector<region_info> get_region_list() const noexcept;

	/// <summary>
	/// 区域互斥.
	/// </summary>
	/// <returns></returns>
	void entry_region_mutex() noexcept;
	void leave_region_mutex() noexcept;

	/// <summary>
	/// 加入区域到尾部.
	/// </summary>
	/// <param name="region">区域相关结构.</param>
	/// <returns></returns>
	void push_region_back(struct region_info& region) noexcept;

	/// <summary>
	/// 从尾部删除区域.
	/// </summary>
	/// <returns></returns>
	void pop_region_back() noexcept;

	/// <summary>
	/// 场景管理函数.
	/// </summary>
	/// <param name="detect">检测物体的结果.</param>
	/// <param name="count">检测物体的数量.</param>
	/// <param name="w">视频帧宽度.</param>
	/// <param name="h">视频帧高度.</param>
	/// <param name="frame">视频帧指针.</param>
	/// <returns></returns>
	void scene_manager(detection* detect, int count, int w, int h, cv::Mat* frame) noexcept;

	/// <summary>
	/// 人流量统计场景.
	/// </summary>
	/// <param name="b">人在视频帧中的位置列表.</param>
	/// <param name="frame">视频帧指针.</param>
	/// <returns></returns>
	void scene_calc_people(std::vector<box> b, cv::Mat* frame) noexcept;

	/// <summary>
	/// 车流量统计场景.
	/// </summary>
	/// <param name="b">车辆在视频帧中的位置列表.</param>
	/// <param name="frame">视频帧指针.</param>
	/// <returns></returns>
	void scene_calc_car(std::vector<box> b, cv::Mat* frame) noexcept;

	/// <summary>
	/// 占用公交车道场景.
	/// </summary>
	/// <param name="b">车辆在视频帧中的位置列表.</param>
	/// <returns></returns>
	void scene_occupy_bus(std::vector<box> b) noexcept;

	/// <summary>
	/// 识别车牌场景.
	/// </summary>
	/// <param name="b">车牌在视频帧中的位置列表.</param>
	/// <param name="frame">视频帧指针.</param>
	/// <returns></returns>
	void scene_recognition_car_id(std::vector<box> b, cv::Mat* frame) noexcept;

	/// <summary>
	/// 获取人流量结构.
	/// </summary>
	/// <returns>返回人流量结构指针</returns>
	struct calc_statistics_info* get_people_info_point() noexcept;

	/// <summary>
	/// 获取车流量结构.
	/// </summary>
	/// <returns>返回车流量结构指针</returns>
	struct calc_statistics_info* get_car_info_point() noexcept;

public:
	/// <summary>
	/// 设置视频路径.
	/// </summary>
	/// <param name="path">视频文件路径.</param>
	/// <returns>设置成功返回true，否则返回false</returns>
	bool set_video_path(const char* path) noexcept;

	/// <summary>
	/// 设置摄像头索引.
	/// </summary>
	/// <param name="index">摄像头索引.</param>
	/// <returns>设置成功返回true，否则返回false</returns>
	bool set_video_index(int index) noexcept;

	/// <summary>
	/// 读取视频帧.
	/// </summary>
	/// <returns>返回一个视频帧指针</returns>
	struct frame_handle* get_video_frame() noexcept;

	/// <summary>
	/// 设置视频帧.
	/// </summary>
	/// <param name="index">视频帧索引.</param>
	/// <param name="rate">视频帧比率.</param>
	/// <returns></returns>
	void set_frame_index(int index) noexcept;
	void set_frame_index(float rate) noexcept;

	/// <summary>
	/// 获取视频播放完成比率.
	/// </summary>
	/// <returns>返回视频播放完成比率</returns>
	float get_finish_rate() noexcept;

	/// <summary>
	/// 获取一帧视频.
	/// </summary>
	/// <param name="path">视频路径.</param>
	/// <param name="index">摄像头索引.</param>
	void get_per_video_frame(const char* path);
	void get_per_video_frame(int index);

public:
	video();
	~video();

	/// <summary>
	/// 开始播放视频.
	/// </summary>
	/// <returns></returns>
	bool start() noexcept;

	/// <summary>
	/// 暂停播放视频.
	/// </summary>
	/// <returns></returns>
	void pause() noexcept;

	/// <summary>
	/// 重新开始播放视频.
	/// </summary>
	/// <returns></returns>
	void restart() noexcept;

	/// <summary>
	/// 关闭视频播放.
	/// </summary>
	/// <returns></returns>
	void close() noexcept;
};
