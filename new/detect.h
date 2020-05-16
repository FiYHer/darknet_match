#pragma once

#include "help.h"
#include "darknet.h"
#include "parser.h"

class detection
{
private:
	//模型路径文件
	char* m_path;

	//标签名称
	char** m_classes;
	int m_classes_count;

	//网络
	network m_net;




public:
	detection();
	~detection();

	//加载模型
	bool load_model() noexcept;

	//卸载模型
	bool unload_model() noexcept;

	//设置模型文件路径
	bool set_model_path(const char* path) noexcept;






};

