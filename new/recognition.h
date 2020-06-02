#pragma once

#include "darknet.h"
#include "parser.h"

#include "help.h"

#include <fstream>
#include <string>

class object_recognition
{
private:
	//�Ƿ��ʼ��ģ��
	bool m_initialize_model;

	//ģ���ļ�·��
	char m_path[max_string_len];

	//����
	network m_net;

	//��ǩ����
	char** m_labels_name;

	//��ǩ����
	int m_labels_count;

public:
	object_recognition() noexcept;
	~object_recognition() noexcept;

	//����ģ���ļ�·��
	bool set_model_path(const char* path) noexcept;

	//����ģ��
	bool load_model() noexcept;

	//ж��ģ��
	bool unload_model() noexcept;
};
