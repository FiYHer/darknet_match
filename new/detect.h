#pragma once

#include "help.h"
#include "darknet.h"
#include "parser.h"

/// <summary>
/// 物体检测类
/// </summary>
class object_detect
{
private:
	//模型路径文件
	char* m_path;

	//标签名称
	char** m_classes_name;
	int m_classes_count;

	//标签颜色
	float** m_classes_color;

	//网络
	network m_net;

	//是否加载
	bool m_loaded;

	//检测相关阈值
	float m_thresh;
	float m_hier_thresh;
	float m_nms;

public:
	object_detect();
	~object_detect();

	/// <summary>
	/// 加载模型.
	/// </summary>
	/// <returns>加载成功返回true，否则返回false</returns>
	bool load_model() noexcept;

	/// <summary>
	/// 卸载模型.
	/// </summary>
	/// <returns>卸载成功返回true，否则返回false</returns>
	bool unload_model() noexcept;

public:
	/// <summary>
	/// 设置模型文件路径.
	/// </summary>
	/// <param name="path">模型文件路径.</param>
	/// <returns>设置成功返回true，否则返回false</returns>
	bool set_model_path(const char* path) noexcept;

	/// <summary>
	/// 获取网络模型.
	/// </summary>
	/// <returns>返回模型指针</returns>
	network* get_network() noexcept;

	/// <summary>
	/// 获取标签数量.
	/// </summary>
	/// <returns>返回类别数量</returns>
	int get_classes_count() const noexcept;

	/// <summary>
	/// 获取标签字符串.
	/// </summary>
	/// <returns>返回类别字符串指针</returns>
	char** get_classes_name() const noexcept;

	/// <summary>
	/// 获取模型是否加载.
	/// </summary>
	/// <returns>返回模型加载的状态标识</returns>
	bool get_model_loader() const noexcept;

	/// <summary>
	/// 获取检测阈值相关.
	/// </summary>
	/// <returns>返回相关阈值</returns>
	float get_thresh() const noexcept;
	float get_hier_thresh() const noexcept;
	float get_nms() const noexcept;

	/// <summary>
	/// 获取类别颜色.
	/// </summary>
	/// <returns>返回类别颜色指针</returns>
	float** get_classes_color() const noexcept;

	/// <summary>
	/// 释放类别名称内存.
	/// </summary>
	/// <returns></returns>
	void free_classes_name() noexcept;

	/// <summary>
	/// 释放类别颜色内存.
	/// </summary>
	/// <returns></returns>
	void free_classes_color() noexcept;
};
