#pragma once
#include <opencv2/opencv.hpp>

//�����������
enum object_classes
{
	object_license_plate,//����
	object_car,//����
	object_bus,//������
	objetc_person,//����
	object_traffic_light//���̵�
};

//��Ƶ��������
enum video_display_mode
{
	//��Ƶģʽ
	e_mode_video,

	//����ͷģʽ
	e_mode_camera
};

//��Ƶ֡����״̬
enum frame_handle_state
{
	//δ����
	e_un_handle,

	//�����
	e_detec_handle,

	//�������
	e_finish_handle
};

//��Ƶ֡������Ϣ
struct frame_handle
{
	//��Ƶ֡
	cv::Mat frame;

	//����״̬
	enum frame_handle_state state;
};

//��������
enum region_type
{
	//��������
	region_bus,

	//������
	region_zebra_cross
};

//����������Ϣ
struct rect_info
{
	//
	int left, top, right, down;

	rect_info() : left(0), top(0), right(0), down(0) {}
	rect_info(int l, int t, int r, int d) : left(l), top(t), right(r), down(d) {}
	rect_info operator=(rect_info& t)
	{
		left = t.left;
		top = t.top;
		right = t.right;
		down = t.down;
		return *this;
	}
};

//������Ϣ
/// <summary>
/// ������Ϣ
/// </summary>
struct region_info
{
	//��������
	struct rect_info rect;

	//��������
	enum region_type type;

	//������ɫ
	unsigned int color;

	region_info() : type(region_bus), color(0) {}
	region_info(struct rect_info r, region_type t, unsigned int c)
	{
		rect = r;
		type = t;
		color = c;
	}
};

//�����������Ϣ
struct calc_people_info
{
	//����
	bool enable;

	//��ǰ������
	unsigned int current_val;

	//���������
	unsigned int max_val;

	calc_people_info() : enable(false) {}
};

struct calc_car_info
{
	//����
	bool enable;

	calc_car_info() : enable(false) {}
};
