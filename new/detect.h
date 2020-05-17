#pragma once

#include "help.h"
#include "darknet.h"
#include "parser.h"

class object_detect
{
private:
	//ģ��·���ļ�
	char* m_path;

	//��ǩ����
	char** m_classes;
	int m_classes_count;

	//����
	network m_net;

	//�Ƿ����
	bool m_loaded;

	float m_thresh;
	float m_hier_thresh;
	float m_nms;

public:
	object_detect();
	~object_detect();

	//����ģ��
	bool load_model() noexcept;

	//ж��ģ��
	bool unload_model() noexcept;

public:
	//����ģ���ļ�·��
	bool set_model_path(const char* path) noexcept;

	//��ȡ����ģ��
	network* get_network() noexcept;

	//��ȡ��ǩ����
	int get_classes_count() const noexcept;

	//��ȡģ���Ƿ����
	bool get_model_loader() const noexcept;

	//��ȡ�����ֵ���
	float get_thresh() const noexcept;
	float get_hier_thresh() const noexcept;
	float get_nms() const noexcept;




};

