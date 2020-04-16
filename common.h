#pragma once

//交通系统标志
#ifndef TRAFFIC_SYSTEM_USE
#define TRAFFIC_SYSTEM_USE
#endif

//C++文件支持
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <map>

//Window文件支持
#include <windows.h>
#include <process.h>
#include <stdio.h>
#include <time.h>
#include <corecrt_io.h>

//d3d文件支持
#include <d3d9.h>
#pragma comment(lib,"d3d9.lib")

//darknet文件支持
#include "list.h"
#include "utils.h"
#include "data.h"
#include "parser.h"
#include "option_list.h"
#include "darknet.h"
#include "image.h"
#include "image_opencv.h"

//imgui文件支持
#include "imgui/imgui.h"
#include "imgui/imgui_impl_win32.h"
#include "imgui/imgui_impl_dx9.h"

//opencv文件支持
#include <opencv2/opencv.hpp>

//默认字符串大小
constexpr int default_char_size = 1024;

//检测对象类型
enum detect_object_type
{
	object_type_car_id,//车牌
	object_type_car,//车辆
	object_type_person,//人类
	object_type_motorbike,//摩托车
	object_type_bicycle,//自行车
	object_type_trafficlight,//红绿灯
	object_type_dog,//狗
	object_type_bus//公交车
};

//对象信息
struct object_info
{
	int left, right, top, down;//位置信息
	int class_index;//所属类
	float confidence;//置信度
};

//颜色信息
struct color_info
{
	//方框颜色
	float box_rgb[3];

	//字体颜色
	float font_rgb[3];

	//厚度
	float thickness;

	color_info() : thickness(1.0f) {}
};

//车辆信息
struct car_info
{
	int car_id[7];//车辆车牌索引

	int times[6];//时间信息
};

//场景相关
struct scene_info
{
	//获取当前分钟
	int get_current_minute()
	{
		time_t timep;
		time(&timep);
		struct tm *prt = gmtime(&timep);
		return prt->tm_min;
	}

	//人数增加
	void increate_human(int value)
	{
		//获取当前是第几分钟
		int current_minute = get_current_minute();

		//不同分钟
		if (current_minute != minute)
		{
			//保存
			minute = current_minute;

			//保存上一分钟的人数
			human_num.push_back(human_minute);

			//只保存十分钟的人数
			int num_size = human_num.size();
			if (num_size > 10) human_num.erase(human_num.begin(), human_num.begin() + (num_size - 10));

			//设置当前分钟的人数
			human_minute = value;
		}
		else human_minute += value;//同一分钟直接递增

		//总人数递增
		human_count += value;
	}

	//增加车流量
	void increate_car(int value)
	{
		//获取当前分钟
		int current_minute = get_current_minute();

		//过了一分钟
		if (current_minute != minute)
		{
			//保存
			minute = current_minute;

			//保存上一分钟车流量
			car_num.push_back(car_minute);

			//只保存十分钟的车流量
			int num_size = car_num.size();
			if (num_size > 10) car_num.erase(car_num.begin(), car_num.begin() + (num_size - 10));

			//保存当前分钟的车数量
			car_minute = value;
		}
		else car_minute += value;
		car_count += value;
	}

	//当前处于第几分钟
	int minute = get_current_minute();

	//人流量
	bool human_traffic;
	unsigned int human_count;//总人流量
	unsigned int human_minute;//当前分钟人流量
	unsigned int human_current;//当前人流量
	unsigned int human_avg;//平均人流量
	std::vector<int> human_num;//每分钟人数

	//车流量
	bool car_traffic;
	unsigned int car_count;//总车流量
	unsigned int car_minute;//当前分钟车流量
	unsigned int car_current;//当前车流量
	unsigned int car_avg;//平均车流量
	std::vector<int> car_num;//每分钟车辆数

	//占用公交车道
	bool occupy_bus_lane;
	std::vector<car_info> occupy_bus_list;//车辆列表

	//闯红灯
	bool rush_red_light;

	//不按导向行驶
	bool not_guided;

	//斑马线不礼让行人
	bool not_zebra_cross;

	scene_info()
	{
		human_count = car_count = 0;
	}
};

//分析相关
struct set_detect_info
{
	double detect_time;//物体检测耗时
	double identify_time;//车牌识别耗时

	float thresh;//阈值
	float hier_thresh;//阈值
	float nms;//非极大值抑制

	set_detect_info() :thresh(0.25f), hier_thresh(0.5f), nms(0.45f) {}
};

//视频控制相关
struct video_control
{
	//是否使用摄像头
	int use_camera;

	//视频路径
	char video_path[default_char_size];

	//摄像头索引
	int camera_index;

	//是否退出线程
	bool leave;

	//检测线程数量
	int detect_count;

	//显示延迟 读取延迟 检测延迟 场景延迟
	int show_delay, read_delay, detect_delay, scene_delay;

	//视频宽度 高度
	int video_size[2];

	video_control() :leave(true), detect_count(2) 
	{
		use_camera = camera_index = 0;
		show_delay = read_delay = detect_delay = scene_delay = 10;
		video_size[0] = video_size[1] = 0;
	}
};

//视频帧相关
struct video_frame_info
{
	//是否正在预测
	bool detecting;

	//是否能够显示
	bool display;

	//原始视频帧
	cv::Mat original_frame;
};

//检测结果
struct detect_result
{
	//检测结果
	detection* data;

	//数量
	int count;

	//宽度 高度
	int width, height;

	detect_result() :data(nullptr), count(0) {}
	detect_result(detection* d, int c, int w, int h) :data(d), count(c), width(w), height(h) {}
	void clear(){ if(data) free_detections(data, count);}
};

//视频处理相关
struct video_handle_info
{
	//是否结束线程
	bool break_state;

	//相关线程是否还在工作
	bool read_frame, detect_frame;

	//视频类指针
	cv::VideoCapture cap;

	//队列最多数量,因为读取速度远远快于检测速度
	int max_frame_count;

	//检测视频帧队列
	std::vector<video_frame_info*> detect_datas;

	//检测结果队列，用于场景检测
	std::vector<detect_result> scene_datas;

	//关键段用于线程同步
	CRITICAL_SECTION critical_srction;

	//各线程延迟设置
	int *show_delay, *read_delay, *detect_delay, *scene_delay;

	//视频宽度 高度
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

//网络层
struct net_info_set
{
	bool initizlie;//是否初始化了
	int classes;//类别数量

	char** classes_name;//标签名字
	struct network this_net;//网络结构
};

//界面显示信息
struct imgui_set
{
	//物体检测模型
	bool show_object_detect_window;

	//车牌识别模型
	bool show_car_id_identify_window;

	//测试图片窗口
	bool show_test_picture_window;

	//测试视频窗口
	bool show_test_video_window;

	//设置马路区域
	bool show_set_load_region_window;
};

//图片相关
struct picture_info
{
	//宽度 高度 通道数
	int w, h, c;

	//图片数据
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

//区域类型
enum region_type
{
	region_zebra_cross,//斑马线
	region_bus_lane,//公交车专用道
	region_street_parking//路边车位
};

//标记相关
struct region_mask
{
	//窗口大小
	ImVec2 window_size;

	//开始位置和结束位置
	ImVec2 pos, size;

	//方框颜色
	ImVec4 rect_color;

	//标记类型
	enum region_type type;

	region_mask() {}
	region_mask(ImVec2 p1, ImVec2 p2, ImVec4 c, region_type t) :pos(p1), size(p2), rect_color(c), type(t) {}
	
	//相对位置
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

//全局设置信息
struct global_set
{
	//窗口类名称
	char window_class_name[default_char_size];

	//窗口句柄
	HWND window_hwnd;

	//d3d9设备相关
	LPDIRECT3D9 direct3d9;
	D3DPRESENT_PARAMETERS d3dpresent;
	LPDIRECT3DDEVICE9 direct3ddevice9;
	LPDIRECT3DTEXTURE9 direct3dtexture9;

	//区域标记相关
	std::vector<region_mask> mask_list;

	//图片设置
	struct picture_info picture_set;

	//视频帧设置
	struct picture_info video_frame_set;

	//界面显示相关
	struct imgui_set imgui_show_set;

	//物体检测网络相关
	struct net_info_set object_detect_net_set;

	//车牌识别网络相关
	struct net_info_set car_id_identify_net;

	//颜色相关
	struct color_info color_set;

	//场景相关
	struct scene_info secne_set;

	//视频检测相关
	struct set_detect_info video_detect_set;
};
extern global_set g_global_set;

//严重错误检查
void check_serious_error(bool state, const char* show_str = "", const char* file_pos = __FILE__, int line_pos = __LINE__);

//显示窗口提示
void show_window_tip(const char* str);

//获取显卡相关信息
cudaDeviceProp* get_gpu_infomation(int gpu_count);

//获取系统类型
int get_os_type();

//获取CPU核心数
int get_cpu_kernel();

//获取物理内存总数
int get_physical_memory();

//选择指定类型的一个文件
bool select_type_file(const char* type_file, char* return_str, int return_str_size = default_char_size);

//读取标签名称
void read_classes_name(std::vector<std::string>& return_data, const char* path);

//初始化物体检测网络
bool initialize_object_detect_net(const char* names_file, const char* cfg_file, const char* weights_file);

//初始化车牌识别网络
bool initialize_car_id_identify_net(const char* names_file, const char* cfg_file, const char* weights_file);

//清理物体检测网络
void clear_object_detect_net();

//清理车牌识别网络
void clear_car_id_identify_net();

//类型判断
inline bool is_object_car_id(int index) { return index == object_type_car_id; }
inline bool is_object_car(int index) { return index == object_type_car; }
inline bool is_object_person(int index) { return index == object_type_person; }
inline bool is_object_motorbike(int index) { return index == object_type_motorbike; }
inline bool is_object_bicycle(int index) { return index == object_type_bicycle; }
inline bool is_object_trafficlight(int index) { return index == object_type_trafficlight; }
inline bool is_object_dog(int index) { return index == object_type_dog; }
inline bool is_object_bus(int index) { return index == object_type_bus; }

//加载图片数据
void read_picture_data(const char* target, image& picture_data, cv::Mat& opencv_data, cv::Mat& rgb_data);

//将Mat数据转化为image数据
void mat_translate_image(const cv::Mat& opencv_data, image& image_data);

//预测图片
void analyse_picture(const char* target, set_detect_info& detect_info, bool show = true);

//预测车牌
double analyse_car_id(cv::Mat& picture_data, box box_info, int* car_id_info);

//验证车牌区域
void check_car_id_rect(cv::Mat roi);

//获取车牌指定位数图像信息
cv::Mat get_car_id_data_from_index(cv::Mat& data, int index);

//获取车牌索引和置信度
void get_max_car_id(float* predictions, int count, int& index, float* confid = nullptr);

//绘制方框和类型
void draw_boxs_and_classes(cv::Mat& picture_data, box box_info, const char* name);

//计算对象位置信息
void get_object_rect(int width, int height, box& box_info, object_info& object);

//绘制方框
void draw_object_rect(cv::Mat& buffer, int left, int top, int right, int down);

//将字符串转化为utf8编码
std::string string_to_utf8(const char* str);

//更新置图片纹理数据
void update_picture_texture(cv::Mat& opencv_data);

//读取一帧视频
void read_video_frame(const char* target);

//分析视频文件
unsigned __stdcall analyse_video(void* prt);

//读取视频帧线程
unsigned __stdcall read_frame_proc(void* prt);

//预测视频帧对象线程
unsigned __stdcall prediction_frame_proc(void* prt);

//场景事件处理
unsigned __stdcall scene_event_proc(void* prt);

//统计人流量
void calc_human_traffic(std::vector<box> b, int width, int height);

//统计车流量
void calc_car_traffic(std::vector<box> b, int width, int height);

//计算实际位置
void calc_trust_box(box& b, int width, int height);

//计算相交
bool calc_intersect(box b1, box b2, float ratio = 0.5f);

//计算是否是同一个
bool calc_same_rect(std::vector<box>& b_list,box& b);

//占用公交车道检测
void check_occupy_bus_lane(std::vector<box> b, int width, int height);







std::vector<std::string> get_path_from_str(const char* str, const char* file_type);
void picture_to_label(const char* path, std::map<std::string,int>& class_names);



