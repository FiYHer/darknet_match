#pragma once

#include "help.h"
#include "darknet.h"
#include "parser.h"

class detection
{
private:
	//ģ��·���ļ�
	char* m_path;

	//��ǩ����
	char** m_classes;
	int m_classes_count;

	//����
	network m_net;




public:
	detection();
	~detection();

	//����ģ��
	bool load_model() noexcept;

	//ж��ģ��
	bool unload_model() noexcept;

	//����ģ���ļ�·��
	bool set_model_path(const char* path) noexcept;






};

