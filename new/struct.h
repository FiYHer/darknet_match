#pragma once
#include <opencv2/opencv.hpp>

enum video_display_mode
{
	//��Ƶģʽ
	e_mode_video,

	//����ͷģʽ
	e_mode_camera
};

enum frame_handle_state
{
	//δ����
	e_un_handle,

	//�����
	e_detec_handle,

	//�������
	e_finish_handle
};

struct frame_handle
{
	//��Ƶ֡
	cv::Mat frame;

	//����״̬
	enum frame_handle_state state;
};





