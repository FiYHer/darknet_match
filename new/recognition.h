#pragma once

#include "darknet.h"
#include "parser.h"

#include "help.h"

#include <fstream>
#include <string>

class object_recognition
{
private:
	//是否初始化模型
	bool m_initialize_model;

	//模型文件路径
	char m_path[max_string_len];

	//网络
	network m_net;

	//标签名称
	char** m_labels_name;

	//标签数量
	int m_labels_count;

public:
	object_recognition() noexcept;
	~object_recognition() noexcept;

	//设置模型文件路径
	bool set_model_path(const char* path) noexcept;

	//加载模型
	bool load_model() noexcept;

	//卸载模型
	bool unload_model() noexcept;
};
