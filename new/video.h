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

	//��ȡһ֡��Ƶͼ��
	bool per_frame();

public:
	//�ж��Ƿ��ȡ��Ƶ֡��
	bool get_is_reading() const noexcept;

	//�ж��Ƿ�����Ƶ֡��
	bool get_is_detecting() const noexcept;
	
	//���ö�ȡ��Ƶ֡״̬
	void set_reading(bool state) noexcept;

	//���ü����Ƶ֡״̬
	void set_detecting(bool state) noexcept;

	//��ȡ��Ƶ�ļ�·��
	const char* get_path() const noexcept;

	//��ȡ����ͷ����
	int get_index() const noexcept;

	//��ȡ��ǰģʽ
	video_display_mode get_mode() const noexcept;

	//��ȡ��Ƶ��ָ��
	cv::VideoCapture* get_capture() noexcept;

	//��ȡ��Ƶ֡�б�ָ��
	std::list<frame_handle*>* get_frames() noexcept;

	//��Ƶ�໥��
	void entry_capture_mutex() noexcept;
	void leave_capture_mutex() noexcept;

	//��Ƶ֡����
	void entry_frame_mutex() noexcept;
	void leave_frame_mutex() noexcept;

	//������ͣ״̬
	void set_payse_state() noexcept;

	//��ȡ��ͣ״̬
	bool get_pause_state() const noexcept;

	//��ȡFPS
	double get_display_fps() const noexcept;

	//��ȡ�����ָ��
	object_detect* get_detect_model() noexcept;

	//��Ƶ֡ת��ͼ��
	image to_image(cv::Mat frame, int out_w, int out_h, int out_c) noexcept;

	//���Ʒ��������
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

	//��ȡ��Ƶ������ɱ���
	float get_finish_rate() noexcept;

	//��ȡһ֡��Ƶ
	void get_per_video_frame(const char* path);
	void get_per_video_frame(int index);


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

