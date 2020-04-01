#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <queue>

#include <windows.h>
#include <process.h>
#include <stdio.h>

#include <d3d9.h>
#include <corecrt_io.h>
#pragma comment(lib,"d3d9.lib")

#include "list.h"
#include "data.h"
#include "parser.h"
#include "option_list.h"
#include "darknet.h"
#include "image.h"
#include "image_opencv.h"

#include <opencv2/opencv.hpp>

//Ĭ���ַ�����С
constexpr int default_char_size = 1024;

//������Ϣ
struct object_info
{
	int left, right, top, down;//λ����Ϣ
	int class_index;//������
	float confidence;//���Ŷ�
};

//��Ƶ֡���
struct video_frame_info
{
	//�Ƿ�����Ԥ��
	bool detecting;

	//�Ƿ��ܹ���ʾ
	bool display;

	//ԭʼ��Ƶ֡
	cv::Mat original_frame;
};

//��Ƶ�������
struct video_handle_info
{
	//��¼�Ƿ�Ҫ�˳���Ƶ��ȡ
	bool break_state;
	bool read_frame;
	bool detect_frame;

	//��Ƶ��ָ��
	cv::VideoCapture cap;

	//�����������,��Ϊ��ȡ�ٶ�ԶԶ���ڼ���ٶ�
	int max_frame_count;

	//�����Ƶ֡����
	std::vector<video_frame_info*> detect_datas;

	//�ؼ��������߳�ͬ��
	CRITICAL_SECTION critical_srction;

	void initialize()
	{ 
		break_state = false;
		read_frame = detect_frame = true;
		max_frame_count = 50;
		InitializeCriticalSection(&critical_srction); 
	}
	void entry() { EnterCriticalSection(&critical_srction); }
	void leave() { LeaveCriticalSection(&critical_srction); }
	void clear() 
	{ 
		break_state = true;
		read_frame = detect_frame = false;
		if (cap.isOpened()) cap.release();
		for (auto& it : detect_datas) delete it;
		detect_datas.clear();
		DeleteCriticalSection(&critical_srction); 
	}
};

//�����
struct net_info_set
{
	bool initizlie;//�Ƿ��ʼ����
	char data_path[default_char_size];//data�ļ�·��
	char names_path[default_char_size];//names�ļ�·��
	char cfg_path[default_char_size];//cfg�ļ�·��
	char weights_path[default_char_size];//Ȩ���ļ�
	int classes;//�������

	char** classes_name;//��ǩ����
	struct network match_net;//����ṹ

};

//������ʾ��Ϣ
struct imgui_set
{
	//�ļ����ô���
	bool show_file_set_window;

	//����ͼƬ����
	bool show_test_picture_window;

	//������Ƶ����
	bool show_test_video_window;

	//��������ͷ����
	bool show_test_camera_window;

};

//ȫ��������Ϣ
struct global_set
{
	//����������
	char window_class_name[1024];

	//���ھ��
	HWND window_hwnd;

	//d3d9�豸���
	LPDIRECT3D9 direct3d9;
	D3DPRESENT_PARAMETERS d3dpresent;
	LPDIRECT3DDEVICE9 direct3ddevice9;
	LPDIRECT3DTEXTURE9 direct3dtexture9;

	//ͼ�����ݣ�������imgui����ʾͼƬ
	int width, height, channel;
	unsigned char* picture_data;

	//��������Ϣ
	int box_number;
	struct detection* detection_data;

	//ͼƬ���ʱ��
	double detection_time;

	//�߳�����
	int video_read_frame_threads;
	int video_detect_frame_threads;

	//�ӳ�����
	int show_video_delay;
	int read_video_delay;
	int detect_video_delay;

	//������ʾ���
	struct imgui_set imgui_show_set;

	//��������
	struct net_info_set net_set;

	global_set()
	{
		//һ����ȡ�߳̾ͱ�4������߳̿���,����������bug
		video_read_frame_threads = 1;
		video_detect_frame_threads = 4;

		show_video_delay = read_video_delay = detect_video_delay = 10;
	}
};
extern global_set g_global_set;

//���ش�����
void check_serious_error(bool state, const char* show_str = "");

//��ʾ������ʾ
void show_window_tip(const char* str);

//ѡ��ָ�����͵�һ���ļ�
bool select_type_file(const char* type_file, char* return_str, int return_str_size = default_char_size);

//��ȡ��ǩ����
void read_classes_name(std::vector<std::string>& return_data, const char* path);

//��ʼ������
bool initialize_net();

//��������
void clear_net();

//����ͼƬ����
void read_picture_data(const char* target, image& picture_data, cv::Mat& opencv_data);

//��Mat����ת��Ϊimage����
void mat_translate_image(const cv::Mat& opencv_data, image& image_data);

//Ԥ��ͼƬ
void analyse_picture(const char* target, std::vector<object_info>& object, int show_type = 0);

//�������λ����Ϣ
void get_object_rect(int width, int height, box& box_info, object_info& object);

//���Ʒ���
void draw_object_rect(cv::Mat& buffer, int left, int top, int right, int down);

//���ַ���ת��Ϊutf8����
std::string string_to_utf8(const char* str);

//������ͼƬ��������
void update_picture_texture(cv::Mat& opencv_data);

//������Ƶ�ļ�
void analyse_video(const char* video_path);

//��ȡ��Ƶ֡�߳�
unsigned __stdcall read_frame_proc(void* prt);

//Ԥ����Ƶ֡�����߳�
unsigned __stdcall prediction_frame_proc(void* prt);





std::vector<std::string> get_path_from_str(const char* str, const char* file_type);
void picture_to_label(const char* path, std::vector<std::string>& class_name, int index);



