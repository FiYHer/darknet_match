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
/// ��Ƶ������
/// </summary>
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

	//����ʶ��ģ��
	object_recognition m_recognition_model;

	//�����б�
	std::vector<region_info> m_regions;

	//���򻥳�
	std::mutex m_region_mutex;

private:
	//���������
	struct calc_statistics_info m_calc_people;

	//��⳵����
	struct calc_statistics_info m_calc_car;

	//����ʶ����
	bool m_recognition_car_id;

private:
	/// <summary>
	/// ��ȡ��Ƶ֡�߳�.
	/// </summary>
	/// <param name="data">����.</param>
	/// <returns></returns>
	static void __cdecl read_frame_thread(void* data);

	/// <summary>
	/// �����Ƶ֡�߳�.
	/// </summary>
	/// <param name="data">����.</param>
	/// <returns></returns>
	static void __cdecl detect_frame_thread(void* data);

	/// <summary>
	/// ����FPS.
	/// </summary>
	/// <returns></returns>
	void update_fps() noexcept;

	/// <summary>
	/// ��ȡһ֡��Ƶͼ��.
	/// </summary>
	/// <returns>��ȡ�ɹ�����true�����򷵻�false</returns>
	bool per_frame();

	/// <summary>
	/// ת��Ϊʵ������.
	/// </summary>
	/// <param name="b">λ�ñ��ʽṹ.</param>
	/// <param name="w">ͼ����.</param>
	/// <param name="h">ͼ��߶�.</param>
	/// <param name="left">���������.</param>
	/// <param name="top">�������ϵ�.</param>
	/// <param name="right">�������ҵ�.</param>
	/// <param name="bot">�������µ�.</param>
	void box_to_pos(box b, int w, int h, int& left, int& top, int& right, int& bot);

	/// <summary>
	/// �ж�����������ཻ��.
	/// </summary>
	/// <param name="a">Aλ�ñ��ʽṹ.</param>
	/// <param name="b">Bλ�ñ��ʽṹ.</param>
	/// <param name="value">�ཻ����.</param>
	/// <returns>
	///   <c>true</c> a��b���ཻ���ʴ���value
	///   <c>false</c> a��b���ཻ����С��value
	/// </returns>
	bool is_coincide_rate(box a, box b, float value = 0.5f);

public:
	/// <summary>
	/// �ж��Ƿ��ȡ��Ƶ֡��.
	/// </summary>
	/// <returns>��Ƶ��ȡ�з���true�����򷵻�false</returns>
	bool get_is_reading() const noexcept;

	/// <summary>
	/// �ж��Ƿ�����Ƶ֡��.
	/// </summary>
	/// <returns>�����Ƶ�з���true�����򷵻�false</returns>
	bool get_is_detecting() const noexcept;

	/// <summary>
	/// ���ö�ȡ��Ƶ֡״̬.
	/// </summary>
	/// <param name="state">���õ�״̬.</param>
	/// <returns></returns>
	void set_reading(bool state) noexcept;

	/// <summary>
	/// ���ü����Ƶ֡״̬.
	/// </summary>
	/// <param name="state">���õ�״̬.</param>
	/// <returns></returns>
	void set_detecting(bool state) noexcept;

	/// <summary>
	/// ��ȡ��Ƶ�ļ�·��.
	/// </summary>
	/// <returns>������Ƶ�ļ��ַ���</returns>
	const char* get_path() const noexcept;

	/// <summary>
	/// ��ȡ����ͷ����.
	/// </summary>
	/// <returns>��������ͷ����</returns>
	int get_index() const noexcept;

	/// <summary>
	/// ��ȡ��ǰģʽ.
	/// </summary>
	/// <returns>���ص�ǰģ��</returns>
	video_display_mode get_mode() const noexcept;

	/// <summary>
	/// ��ȡ��Ƶ��ָ��.
	/// </summary>
	/// <returns>������Ƶ��ȡ��ָ��</returns>
	cv::VideoCapture* get_capture() noexcept;

	/// <summary>
	/// ��ȡ��Ƶ֡�б�ָ��.
	/// </summary>
	/// <returns>������Ƶ֡�б��ָ��</returns>
	std::list<frame_handle*>* get_frames() noexcept;

	/// <summary>
	/// ��Ƶ�໥��.
	/// </summary>
	/// <returns></returns>
	void entry_capture_mutex() noexcept;
	void leave_capture_mutex() noexcept;

	/// <summary>
	/// ��Ƶ֡����.
	/// </summary>
	/// <returns></returns>
	void entry_frame_mutex() noexcept;
	void leave_frame_mutex() noexcept;

	/// <summary>
	/// ������ͣ״̬.
	/// </summary>
	/// <returns></returns>
	void set_payse_state() noexcept;

	/// <summary>
	/// ��ȡ��ͣ״̬.
	/// </summary>
	/// <returns>������ͣ״̬</returns>
	bool get_pause_state() const noexcept;

	/// <summary>
	/// ��ȡFPS.
	/// </summary>
	/// <returns>���ص�ǰ��fps</returns>
	double get_display_fps() const noexcept;

	/// <summary>
	/// ��ȡ�����ָ��.
	/// </summary>
	/// <returns>����������ģ��ָ��</returns>
	object_detect* get_detect_model() noexcept;

	/// <summary>
	/// ��ȡʶ����ָ��.
	/// </summary>
	/// <returns>���س���ʶ��ģ��ָ��</returns>
	object_recognition* get_recognition_model() noexcept;

	/// <summary>
	/// ��Ƶ֡cv::Matת��imageͼ��.
	/// </summary>
	/// <param name="frame">��Ƶ֡.</param>
	/// <param name="out_w">���.</param>
	/// <param name="out_h">�߶�.</param>
	/// <param name="out_c">ͨ����.</param>
	/// <returns>����ת�����image</returns>
	image to_image(cv::Mat frame, int out_w, int out_h, int out_c) noexcept;

	/// <summary>
	/// ���Ʒ��������.
	/// </summary>
	/// <param name="detect">The detect.</param>
	/// <param name="count">The count.</param>
	/// <param name="frame">The frame.</param>
	/// <returns></returns>
	void draw_box_and_font(detection* detect, int count, cv::Mat* frame) noexcept;

	/// <summary>
	/// �������úõ��������.
	/// </summary>
	/// <param name="frame">��Ƶ֡.</param>
	/// <returns></returns>
	void draw_regions(cv::Mat* frame) noexcept;

	/// <summary>
	/// ��ȡ�����б�.
	/// </summary>
	/// <returns>��������ṹ�б�</returns>
	std::vector<region_info> get_region_list() const noexcept;

	/// <summary>
	/// ���򻥳�.
	/// </summary>
	/// <returns></returns>
	void entry_region_mutex() noexcept;
	void leave_region_mutex() noexcept;

	/// <summary>
	/// ��������β��.
	/// </summary>
	/// <param name="region">������ؽṹ.</param>
	/// <returns></returns>
	void push_region_back(struct region_info& region) noexcept;

	/// <summary>
	/// ��β��ɾ������.
	/// </summary>
	/// <returns></returns>
	void pop_region_back() noexcept;

	/// <summary>
	/// ����������.
	/// </summary>
	/// <param name="detect">�������Ľ��.</param>
	/// <param name="count">������������.</param>
	/// <param name="w">��Ƶ֡���.</param>
	/// <param name="h">��Ƶ֡�߶�.</param>
	/// <param name="frame">��Ƶָ֡��.</param>
	/// <returns></returns>
	void scene_manager(detection* detect, int count, int w, int h, cv::Mat* frame) noexcept;

	/// <summary>
	/// ������ͳ�Ƴ���.
	/// </summary>
	/// <param name="b">������Ƶ֡�е�λ���б�.</param>
	/// <param name="frame">��Ƶָ֡��.</param>
	/// <returns></returns>
	void scene_calc_people(std::vector<box> b, cv::Mat* frame) noexcept;

	/// <summary>
	/// ������ͳ�Ƴ���.
	/// </summary>
	/// <param name="b">��������Ƶ֡�е�λ���б�.</param>
	/// <param name="frame">��Ƶָ֡��.</param>
	/// <returns></returns>
	void scene_calc_car(std::vector<box> b, cv::Mat* frame) noexcept;

	/// <summary>
	/// ռ�ù�����������.
	/// </summary>
	/// <param name="b">��������Ƶ֡�е�λ���б�.</param>
	/// <returns></returns>
	void scene_occupy_bus(std::vector<box> b) noexcept;

	/// <summary>
	/// ʶ���Ƴ���.
	/// </summary>
	/// <param name="b">��������Ƶ֡�е�λ���б�.</param>
	/// <param name="frame">��Ƶָ֡��.</param>
	/// <returns></returns>
	void scene_recognition_car_id(std::vector<box> b, cv::Mat* frame) noexcept;

	/// <summary>
	/// ��ȡ�������ṹ.
	/// </summary>
	/// <returns>�����������ṹָ��</returns>
	struct calc_statistics_info* get_people_info_point() noexcept;

	/// <summary>
	/// ��ȡ�������ṹ.
	/// </summary>
	/// <returns>���س������ṹָ��</returns>
	struct calc_statistics_info* get_car_info_point() noexcept;

public:
	/// <summary>
	/// ������Ƶ·��.
	/// </summary>
	/// <param name="path">��Ƶ�ļ�·��.</param>
	/// <returns>���óɹ�����true�����򷵻�false</returns>
	bool set_video_path(const char* path) noexcept;

	/// <summary>
	/// ��������ͷ����.
	/// </summary>
	/// <param name="index">����ͷ����.</param>
	/// <returns>���óɹ�����true�����򷵻�false</returns>
	bool set_video_index(int index) noexcept;

	/// <summary>
	/// ��ȡ��Ƶ֡.
	/// </summary>
	/// <returns>����һ����Ƶָ֡��</returns>
	struct frame_handle* get_video_frame() noexcept;

	/// <summary>
	/// ������Ƶ֡.
	/// </summary>
	/// <param name="index">��Ƶ֡����.</param>
	/// <param name="rate">��Ƶ֡����.</param>
	/// <returns></returns>
	void set_frame_index(int index) noexcept;
	void set_frame_index(float rate) noexcept;

	/// <summary>
	/// ��ȡ��Ƶ������ɱ���.
	/// </summary>
	/// <returns>������Ƶ������ɱ���</returns>
	float get_finish_rate() noexcept;

	/// <summary>
	/// ��ȡһ֡��Ƶ.
	/// </summary>
	/// <param name="path">��Ƶ·��.</param>
	/// <param name="index">����ͷ����.</param>
	void get_per_video_frame(const char* path);
	void get_per_video_frame(int index);

public:
	video();
	~video();

	/// <summary>
	/// ��ʼ������Ƶ.
	/// </summary>
	/// <returns></returns>
	bool start() noexcept;

	/// <summary>
	/// ��ͣ������Ƶ.
	/// </summary>
	/// <returns></returns>
	void pause() noexcept;

	/// <summary>
	/// ���¿�ʼ������Ƶ.
	/// </summary>
	/// <returns></returns>
	void restart() noexcept;

	/// <summary>
	/// �ر���Ƶ����.
	/// </summary>
	/// <returns></returns>
	void close() noexcept;
};
