#pragma once
#include <opencv2/opencv.hpp>

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

struct region_info
{
	//
	int x, y, w, h;

	//��������
	enum region_type type;


};














