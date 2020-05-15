#pragma once

#include "image_opencv.h"
#include "opencv2/opencv.hpp"

#include "struct.h"
#include "help.h"

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
	const int m_detect_count = 2;

private:
	//视频文件路径
	char* m_path;
	
	//摄像头索引
	int m_index;

	//播放模式
	enum video_display_mode m_mode;

	//视频播放
	cv::VideoCapture m_capture;

	//视频帧
	std::list<frame_handle*> m_frames;

	//视频帧互斥
	std::mutex m_frame_mutex;

	//暂停视频播放
	bool m_pause_video;

private:
	//读取视频帧线程
	static void __cdecl read_frame_thread(void* data);

	//检测视频帧线程
	static void __cdecl detect_frame_thread(void* data);

public:
	bool get_is_reading() const noexcept;
	bool get_is_detecting() const noexcept;

	void set_reading(bool state) noexcept;
	void set_detecting(bool state) noexcept;

	const char* get_path() const noexcept;
	int get_index() const noexcept;

	video_display_mode get_mode() const noexcept;
	cv::VideoCapture* get_capture() noexcept;
	std::list<frame_handle*>* get_frames() noexcept;

	void entry_mutex() noexcept;
	void leave_mutex() noexcept;

	bool get_pause_state() const noexcept;

public:
	//设置视频路径
	bool set_video_path(const char* path) noexcept;

	//设置摄像头索引
	bool set_video_index(int index) noexcept;

	//读取视频帧
	struct frame_handle* get_video_frame() noexcept;

	//设置视频帧
	void set_frame_index() noexcept;

	//获取视频播放比率
	float get_finish_rate() noexcept;

public:
	video();
	~video();

	//开始播放视频
	void start() noexcept;
	
	//暂停播放视频
	void pause() noexcept;

	//重新开始播放视频
	void restart() noexcept;

	//关闭视频播放
	void close() noexcept;

};

