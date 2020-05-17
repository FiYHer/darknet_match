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
	//���ڶ�ȡ��Ƶ֡
	bool m_reading;

	//���ڼ����Ƶ֡
	bool m_detecting;

	//��Ƶ�ļ�·��
	char* m_path;
	
	//����ͷ����
	int m_index;

	//����ģʽ
	enum video_display_mode m_mode;

	//��Ƶ����
	cv::VideoCapture m_capture;

	//��Ƶ���Ż���
	std::mutex m_capture_mutex;

	//��Ƶ֡
	std::list<frame_handle*> m_frames;

	//��Ƶ֡����
	std::mutex m_frame_mutex;

	//��ͣ��Ƶ����
	bool m_pause_video;

	//fps
	double m_display_fps;

	//������ģ��
	object_detect m_detect_model;


private:
	//��ȡ��Ƶ֡�߳�
	static void __cdecl read_frame_thread(void* data);

	//�����Ƶ֡�߳�
	static void __cdecl detect_frame_thread(void* data);

	//����FPS
	void update_fps() noexcept;

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

	void entry_capture_mutex() noexcept;
	void leave_capture_mutex() noexcept;

	void entry_frame_mutex() noexcept;
	void leave_frame_mutex() noexcept;

	bool get_pause_state() const noexcept;

	double get_display_fps() const noexcept;

	object_detect* get_detect_model() noexcept;

	image to_image(cv::Mat frame, int out_w, int out_h, int out_c) noexcept;

	void draw_box_and_font(detection* detect, int count, cv::Mat* frame) noexcept;


public:
	//������Ƶ·��
	bool set_video_path(const char* path) noexcept;

	//��������ͷ����
	bool set_video_index(int index) noexcept;

	//��ȡ��Ƶ֡
	struct frame_handle* get_video_frame() noexcept;

	//������Ƶ֡
	void set_frame_index(int index) noexcept;
	void set_frame_index(float rate) noexcept;

	//��ȡ��Ƶ���ű���
	float get_finish_rate() noexcept;

public:
	video();
	~video();

	//��ʼ������Ƶ
	bool start() noexcept;
	
	//��ͣ������Ƶ
	void pause() noexcept;

	//���¿�ʼ������Ƶ
	void restart() noexcept;

	//�ر���Ƶ����
	void close() noexcept;

};

