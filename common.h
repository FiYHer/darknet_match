#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <map>

#include <windows.h>
#include <process.h>
#include <stdio.h>
#include <time.h>

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

#include "imgui/imgui.h"
#include "imgui/imgui_impl_win32.h"
#include "imgui/imgui_impl_dx9.h"

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

//��ɫ��Ϣ
struct color_info
{
	//������ɫ
	float box_rgb[3];

	//������ɫ
	float font_rgb[3];

	//���
	float thickness;

	color_info() : thickness(1.0f) {}
};

//�������
struct scene_info
{
	//��ʼʱ��

	//������
	bool human_traffic;
	unsigned int human_count;

	//������
	bool car_traffic;
	unsigned int car_count;

	//ռ�ù�������
	bool occupy_bus_lane;

	//�����
	bool rush_red_light;

	//����������ʻ
	bool not_guided;

	//�����߲���������
	bool not_zebra_cross;

	scene_info()
	{
		human_count = car_count = 0;
	}
};

//�������
struct set_detect_info
{
	double detect_time;//����ʱ

	float thresh;//��ֵ
	float hier_thresh;//��ֵ
	float nms;//�Ǽ���ֵ����

	set_detect_info() :thresh(0.25f), hier_thresh(0.5f), nms(0.45f) {}
};

//��Ƶ�������
struct video_control
{
	//�Ƿ�ʹ������ͷ
	int use_camera;

	//��Ƶ·��
	char video_path[default_char_size];

	//����ͷ����
	int camera_index;

	//�Ƿ��˳��߳�
	bool leave;

	//����߳�����
	int detect_count;

	//��ʾ�ӳ� ��ȡ�ӳ� ����ӳ� �����ӳ�
	int show_delay, read_delay, detect_delay, scene_delay;

	video_control() :leave(true), detect_count(6) 
	{
		use_camera = camera_index = 0;
		show_delay = read_delay = detect_delay = scene_delay = 10;
	}
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

//�����
struct detect_result
{
	//�����
	detection* data;

	//����
	int count;

	//��� �߶�
	int width, height;

	detect_result() :data(nullptr), count(0) {}
	detect_result(detection* d, int c, int w, int h) :data(d), count(c), width(w), height(h) {}
	void clear(){ if(data) free_detections(data, count);}
};

//��Ƶ�������
struct video_handle_info
{
	//�Ƿ�����߳�
	bool break_state;

	//����߳��Ƿ��ڹ���
	bool read_frame, detect_frame;

	//��Ƶ��ָ��
	cv::VideoCapture cap;

	//�����������,��Ϊ��ȡ�ٶ�ԶԶ���ڼ���ٶ�
	int max_frame_count;

	//�����Ƶ֡����
	std::vector<video_frame_info*> detect_datas;

	//��������У����ڳ������
	std::vector<detect_result> scene_datas;

	//�ؼ��������߳�ͬ��
	CRITICAL_SECTION critical_srction;

	//���߳��ӳ�����
	int *show_delay, *read_delay, *detect_delay, *scene_delay;

	void initialize()
	{ 
		break_state = false;
		read_frame = detect_frame = true;
		max_frame_count = 50;
		show_delay = read_delay = detect_delay = scene_delay = nullptr;
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

	//������·����
	bool show_set_load_region_window;
};

//ͼƬ���
struct picture_info
{
	//��� �߶� ͨ����
	int w, h, c;

	//ͼƬ����
	unsigned char* data;

	picture_info() :w(0), h(0), c(0), data(nullptr) {}
	void make(int _w, int _h, int _c)
	{
		clear();
		w = _w;
		h = _h;
		c = _c;
		data = new unsigned char[w * h * c];
		assert(data);
	}
	void clear() 
	{
		if (data) delete[] data;
		data = nullptr;
		w = h = c = 0;
	}
};

//��������
enum region_type
{
	region_zebra_cross,//������
	region_bus_lane,//������ר�õ�
	region_street_parking//·�߳�λ
};

//������
struct region_mask
{
	//��ʼλ�úͽ���λ��
	ImVec2 start_pos, stop_pos;

	//������ɫ
	ImVec4 rect_color;

	//�������
	enum region_type type;

	region_mask() {}
	region_mask(ImVec2 p1, ImVec2 p2, ImVec4 c, region_type t) :start_pos(p1), stop_pos(p2), rect_color(c), type(t) {}
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

	//����fps
	double fps[2];

	//ͼƬ����
	struct picture_info picture_set;

	//��Ƶ֡����
	struct picture_info video_frame_set;

	//������ʾ���
	struct imgui_set imgui_show_set;

	//��������
	struct net_info_set net_set;

	//��ɫ���
	struct color_info color_set;

	//������
	std::vector<region_mask> mask_list;

	//�������
	scene_info secne_set;

	//��Ƶ������
	struct set_detect_info video_detect_set;
};
extern global_set g_global_set;

//���ش�����
void check_serious_error(bool state, const char* show_str = "");

//��ʾ������ʾ
void show_window_tip(const char* str);

//��ȡ�Կ�������
int get_gpu_count();

//��ȡ�Կ������Ϣ
cudaDeviceProp* get_gpu_infomation(int gpu_count);

//��ȡϵͳ����
int get_os_type();

//��ȡCPU������
int get_cpu_kernel();

//��ȡ�����ڴ�����
int get_physical_memory();

//ѡ��ָ�����͵�һ���ļ�
bool select_type_file(const char* type_file, char* return_str, int return_str_size = default_char_size);

//��ȡ��ǩ����
void read_classes_name(std::vector<std::string>& return_data, const char* path);

//��ʼ������
bool initialize_net(const char* names_file, const char* cfg_file, const char* weights_file);

//��������
void clear_net();

//����ͼƬ����
void read_picture_data(const char* target, image& picture_data, cv::Mat& opencv_data, cv::Mat& rgb_data);

//��Mat����ת��Ϊimage����
void mat_translate_image(const cv::Mat& opencv_data, image& image_data);

//Ԥ��ͼƬ
void analyse_picture(const char* target, set_detect_info& detect_info, bool show = false);

//���Ʒ��������
void draw_boxs_and_classes(cv::Mat& picture_data, box box_info, const char* name);

//�������λ����Ϣ
void get_object_rect(int width, int height, box& box_info, object_info& object);

//���Ʒ���
void draw_object_rect(cv::Mat& buffer, int left, int top, int right, int down);

//���ַ���ת��Ϊutf8����
std::string string_to_utf8(const char* str);

//������ͼƬ��������
void update_picture_texture(cv::Mat& opencv_data);

//��ȡһ֡��Ƶ
void read_video_frame(const char* target);

//������Ƶ�ļ�
unsigned __stdcall analyse_video(void* prt);

//��ȡ��Ƶ֡�߳�
unsigned __stdcall read_frame_proc(void* prt);

//Ԥ����Ƶ֡�����߳�
unsigned __stdcall prediction_frame_proc(void* prt);

//�����¼�����
unsigned __stdcall scene_event_proc(void* prt);

//ͳ��������
void calc_human_traffic(std::vector<box>& b, int width, int height);

//ͳ�Ƴ�����
void calc_car_traffic(std::vector<box>& b, int width, int height);

//����ʵ��λ��
void calc_trust_box(box& b, int width, int height);

//�����Ƿ���ͬһ��
bool calc_same_rect(std::vector<box>& b_list,box& b);



std::vector<std::string> get_path_from_str(const char* str, const char* file_type);
void picture_to_label(const char* path, std::map<std::string,int>& class_names);



