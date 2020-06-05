#pragma once

#include "darknet.h"

#include <opencv2/opencv.hpp>
#include "help.h"

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
	//���ڿ��
	float w, h;

	//
	float left, top, right, down;

	rect_info() : left(0), top(0), right(0), down(0) {}
	rect_info(float l, float t, float r, float d) : left(l), top(t), right(r), down(d) {}
	rect_info operator=(rect_info& t)
	{
		left = t.left;
		top = t.top;
		right = t.right;
		down = t.down;
		return *this;
	}
};

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

	//���ع淶����λ����Ϣ
	box to_box_data() noexcept
	{
		box b;
		b.x = (rect.left - 8.0f) / (rect.w - 8.0f);
		b.y = (rect.top - 35.0f) / (rect.h - 35.0f);
		b.w = (rect.right - 8.0f) / (rect.w - 8.0f);
		b.h = (rect.down - 35.0f) / (rect.h - 35.0f);
		return b;
	}

	region_info() : type(region_bus), color(0) {}
	region_info(struct rect_info r, region_type t, unsigned int c)
	{
		rect = r;
		type = t;
		color = c;
	}
};

//�����/��������Ϣ
struct calc_statistics_info
{
	//����
	bool enable;

	//��ǰ��/������
	unsigned int current_val;

	//�����/������
	unsigned int max_val;

	//��ǰʱ������һ��ʱ��
	int current_time, last_time;

	//��¼ÿ���ӵ���/������
	std::vector<unsigned int> val_list;

	//��¼��/��ͼ��֡
	std::vector<cv::Mat> val_images;
	std::mutex image_mutex;

	//���µ�ǰ��/������
	void update_current_val(unsigned int val) noexcept { current_val = val; if (val > max_val) max_val = val; }

	//���¸ռ�⵽����/������
	void update_new_val(unsigned int val) noexcept
	{
		current_time = get_current_minute();

		if (last_time != current_time)
		{
			last_time = current_time;
			val_list.push_back(val);
		}
		else
		{
			int index = val_list.size() - 1;
			if (index == -1) val_list.push_back(val);
			else val_list[index] += val;
		}
	}

	//������/����Ƶ֡
	void update_image(cv::Mat frame) noexcept
	{
		const cv::Size size{ 100,100 };
		cv::resize(frame, frame, size);

		entry_image();
		val_images.push_back(std::move(frame));
		while (val_images.size() > 5)
		{
			val_images.begin()->release();
			val_images.erase(val_images.begin());
		}
		leave_image();
	}

	void entry_image() noexcept { image_mutex.lock(); }
	void leave_image() noexcept { image_mutex.unlock(); }

	//�������
	void clear() noexcept
	{
		current_val = max_val = 0;
		val_list.clear();

		entry_image();
		for (auto& it : val_images) it.release();
		val_images.clear();
		leave_image();
	}

	calc_statistics_info() : enable(false)
	{
		clear();
		last_time = get_current_minute();
	}
};
