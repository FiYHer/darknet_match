#pragma once

//��ͨϵͳ��־
#ifndef TRAFFIC_SYSTEM_USE
#define TRAFFIC_SYSTEM_USE
#endif

//C++�ļ�֧��
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <map>

//Window�ļ�֧��
#include <windows.h>
#include <process.h>
#include <stdio.h>
#include <time.h>
#include <corecrt_io.h>

//d3d�ļ�֧��
#include <d3d9.h>
#pragma comment(lib,"d3d9.lib")

//darknet�ļ�֧��
#include "list.h"
#include "utils.h"
#include "data.h"
#include "parser.h"
#include "option_list.h"
#include "darknet.h"
#include "image.h"
#include "image_opencv.h"

//imgui�ļ�֧��
#include "imgui/imgui.h"
#include "imgui/imgui_impl_win32.h"
#include "imgui/imgui_impl_dx9.h"

//opencv�ļ�֧��
#include <opencv2/opencv.hpp>

//Ĭ���ַ�����С
constexpr int default_char_size = 1024;

//����������
enum detect_object_type
{
	object_type_car_id,//����
	object_type_car,//����
	object_type_person,//����
	object_type_motorbike,//Ħ�г�
	object_type_bicycle,//���г�
	object_type_trafficlight,//���̵�
	object_type_dog,//��
	object_type_bus//������
};

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

//������Ϣ
struct car_info
{
	int car_id[7];//������������

	int times[6];//ʱ����Ϣ
};

//�������
struct scene_info
{
	//��ȡ��ǰ����
	int get_current_minute()
	{
		time_t timep;
		time(&timep);
		struct tm *prt = gmtime(&timep);
		return prt->tm_min;
	}

	//��������
	void increate_human(int value)
	{
		//��ȡ��ǰ�ǵڼ�����
		int current_minute = get_current_minute();

		//��ͬ����
		if (current_minute != minute)
		{
			//����
			minute = current_minute;

			//������һ���ӵ�����
			human_num.push_back(human_minute);

			//ֻ����ʮ���ӵ�����
			int num_size = human_num.size();
			if (num_size > 10) human_num.erase(human_num.begin(), human_num.begin() + (num_size - 10));

			//���õ�ǰ���ӵ�����
			human_minute = value;
		}
		else human_minute += value;//ͬһ����ֱ�ӵ���

		//����������
		human_count += value;
	}

	//���ӳ�����
	void increate_car(int value)
	{
		//��ȡ��ǰ����
		int current_minute = get_current_minute();

		//����һ����
		if (current_minute != minute)
		{
			//����
			minute = current_minute;

			//������һ���ӳ�����
			car_num.push_back(car_minute);

			//ֻ����ʮ���ӵĳ�����
			int num_size = car_num.size();
			if (num_size > 10) car_num.erase(car_num.begin(), car_num.begin() + (num_size - 10));

			//���浱ǰ���ӵĳ�����
			car_minute = value;
		}
		else car_minute += value;
		car_count += value;
	}

	//��ǰ���ڵڼ�����
	int minute = get_current_minute();

	//������
	bool human_traffic;
	unsigned int human_count;//��������
	unsigned int human_minute;//��ǰ����������
	unsigned int human_current;//��ǰ������
	unsigned int human_avg;//ƽ��������
	std::vector<int> human_num;//ÿ��������

	//������
	bool car_traffic;
	unsigned int car_count;//�ܳ�����
	unsigned int car_minute;//��ǰ���ӳ�����
	unsigned int car_current;//��ǰ������
	unsigned int car_avg;//ƽ��������
	std::vector<int> car_num;//ÿ���ӳ�����

	//ռ�ù�������
	bool occupy_bus_lane;
	std::vector<car_info> occupy_bus_list;//�����б�

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
	double detect_time;//�������ʱ
	double identify_time;//����ʶ���ʱ

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

	//��Ƶ��� �߶�
	int video_size[2];

	video_control() :leave(true), detect_count(2) 
	{
		use_camera = camera_index = 0;
		show_delay = read_delay = detect_delay = scene_delay = 10;
		video_size[0] = video_size[1] = 0;
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

	//��Ƶ��� �߶�
	int video_width, video_height;

	void initialize()
	{ 
		break_state = false;
		read_frame = detect_frame = true;
		max_frame_count = 50;
		show_delay = read_delay = detect_delay = scene_delay = nullptr;
		video_width = video_height = 0;
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
	struct network this_net;//����ṹ
};

//������ʾ��Ϣ
struct imgui_set
{
	//������ģ��
	bool show_object_detect_window;

	//����ʶ��ģ��
	bool show_car_id_identify_window;

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
	//���ڴ�С
	ImVec2 window_size;

	//��ʼλ�úͽ���λ��
	ImVec2 pos, size;

	//������ɫ
	ImVec4 rect_color;

	//�������
	enum region_type type;

	region_mask() {}
	region_mask(ImVec2 p1, ImVec2 p2, ImVec4 c, region_type t) :pos(p1), size(p2), rect_color(c), type(t) {}
	
	//���λ��
	box get_box()
	{
		float w_scale = 1.0f / window_size.x;
		float h_scale = 1.0f / window_size.y;

		float p_x = pos.x - 8.0f;
		float p_y = pos.y - 50.0f;
		float s_x = size.x - 8.0f;
		float s_y = size.y - 50.0f;

		float x_center = (p_x+ s_x) / 2.0f - 1.0f;
		float y_center = (p_y + s_y) / 2.0f - 1.0f;
		float width_scale = abs(p_x - s_x);
		float height_scale = abs(p_y - s_y);

		x_center *= w_scale;
		y_center *= h_scale;
		width_scale *= w_scale;
		height_scale *= h_scale;

		box this_box;
		this_box.x = x_center;
		this_box.y = y_center;
		this_box.w = width_scale;
		this_box.h = height_scale;
		return this_box;
	}
};

//ȫ��������Ϣ
struct global_set
{
	//����������
	char window_class_name[default_char_size];

	//���ھ��
	HWND window_hwnd;

	//d3d9�豸���
	LPDIRECT3D9 direct3d9;
	D3DPRESENT_PARAMETERS d3dpresent;
	LPDIRECT3DDEVICE9 direct3ddevice9;
	LPDIRECT3DTEXTURE9 direct3dtexture9;

	//���������
	std::vector<region_mask> mask_list;

	//ͼƬ����
	struct picture_info picture_set;

	//��Ƶ֡����
	struct picture_info video_frame_set;

	//������ʾ���
	struct imgui_set imgui_show_set;

	//�������������
	struct net_info_set object_detect_net_set;

	//����ʶ���������
	struct net_info_set car_id_identify_net;

	//��ɫ���
	struct color_info color_set;

	//�������
	struct scene_info secne_set;

	//��Ƶ������
	struct set_detect_info video_detect_set;
};
extern global_set g_global_set;

//���ش�����
void check_serious_error(bool state, const char* show_str = "", const char* file_pos = __FILE__, int line_pos = __LINE__);

//��ʾ������ʾ
void show_window_tip(const char* str);

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

//��ʼ������������
bool initialize_object_detect_net(const char* names_file, const char* cfg_file, const char* weights_file);

//��ʼ������ʶ������
bool initialize_car_id_identify_net(const char* names_file, const char* cfg_file, const char* weights_file);

//��������������
void clear_object_detect_net();

//������ʶ������
void clear_car_id_identify_net();

//�����ж�
inline bool is_object_car_id(int index) { return index == object_type_car_id; }
inline bool is_object_car(int index) { return index == object_type_car; }
inline bool is_object_person(int index) { return index == object_type_person; }
inline bool is_object_motorbike(int index) { return index == object_type_motorbike; }
inline bool is_object_bicycle(int index) { return index == object_type_bicycle; }
inline bool is_object_trafficlight(int index) { return index == object_type_trafficlight; }
inline bool is_object_dog(int index) { return index == object_type_dog; }
inline bool is_object_bus(int index) { return index == object_type_bus; }

//����ͼƬ����
void read_picture_data(const char* target, image& picture_data, cv::Mat& opencv_data, cv::Mat& rgb_data);

//��Mat����ת��Ϊimage����
void mat_translate_image(const cv::Mat& opencv_data, image& image_data);

//Ԥ��ͼƬ
void analyse_picture(const char* target, set_detect_info& detect_info, bool show = true);

//Ԥ�⳵��
double analyse_car_id(cv::Mat& picture_data, box box_info, int* car_id_info);

//��֤��������
void check_car_id_rect(cv::Mat roi);

//��ȡ����ָ��λ��ͼ����Ϣ
cv::Mat get_car_id_data_from_index(cv::Mat& data, int index);

//��ȡ�������������Ŷ�
void get_max_car_id(float* predictions, int count, int& index, float* confid = nullptr);

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
void calc_human_traffic(std::vector<box> b, int width, int height);

//ͳ�Ƴ�����
void calc_car_traffic(std::vector<box> b, int width, int height);

//����ʵ��λ��
void calc_trust_box(box& b, int width, int height);

//�����ཻ
bool calc_intersect(box b1, box b2, float ratio = 0.5f);

//�����Ƿ���ͬһ��
bool calc_same_rect(std::vector<box>& b_list,box& b);

//ռ�ù����������
void check_occupy_bus_lane(std::vector<box> b, int width, int height);







std::vector<std::string> get_path_from_str(const char* str, const char* file_type);
void picture_to_label(const char* path, std::map<std::string,int>& class_names);



