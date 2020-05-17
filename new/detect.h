#pragma once

#include "help.h"
#include "darknet.h"
#include "parser.h"

class object_detect
{
private:
	//模型路径文件
	char* m_path;

	//标签名称
	char** m_classes;
	int m_classes_count;

	//网络
	network m_net;

	//是否加载
	bool m_loaded;

	float m_thresh;
	float m_hier_thresh;
	float m_nms;

public:
	object_detect();
	~object_detect();

	//加载模型
	bool load_model() noexcept;

	//卸载模型
	bool unload_model() noexcept;

public:
	//设置模型文件路径
	bool set_model_path(const char* path) noexcept;

	//获取网络模型
	network* get_network() noexcept;

	//获取标签数量
	int get_classes_count() const noexcept;

	//获取模型是否加载
	bool get_model_loader() const noexcept;

	//获取检测阈值相关
	float get_thresh() const noexcept;
	float get_hier_thresh() const noexcept;
	float get_nms() const noexcept;




};

