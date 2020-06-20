#pragma once

#include "darknet.h"
#include "parser.h"
#include "opencv2/opencv.hpp"

#include "help.h"

#include <fstream>
#include <string>
#include <vector>

/// <summary>
/// 车牌号
/// </summary>
static const char* car_ids[] = { "0",
"1",
"2",
"3",
"4",
"5",
"6",
"7",
"8",
"9",
"A",
"B",
"C",
"D",
"E",
"F",
"G",
"H",
"J",
"K",
"L",
"M",
"N",
"O",
"P",
"Q",
"R",
"S",
"T",
"U",
"V",
"W",
"X",
"Y",
"Z",
"皖",
"沪",
"津",
"渝",
"冀",
"晋",
"蒙",
"辽",
"吉",
"黑",
"苏",
"浙",
"京",
"闽",
"赣",
"鲁",
"豫",
"鄂",
"湘",
"粤",
"桂",
"琼",
"川",
"贵",
"云",
"藏",
"陕",
"甘",
"青",
"宁",
"新",
"警",
"学" };

/// <summary>
/// 车牌识别类
/// </summary>
class object_recognition
{
private:
	//是否初始化模型
	bool m_initialize_model;

	//模型文件路径
	char m_path[max_string_len];

	//识别网络
	network m_net;

	//标签名称
	char** m_labels_name;

	//标签数量
	int m_labels_count;

public:
	object_recognition() noexcept;
	~object_recognition() noexcept;

	/// <summary>
	/// 设置模型文件路径.
	/// </summary>
	/// <param name="path">模型文件路径.</param>
	/// <returns>设置成功返回true，否则返回false</returns>
	bool set_model_path(const char* path) noexcept;

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

	/// <summary>
	/// 判断是否加载模型.
	/// </summary>
	/// <returns>模型加载了返回true，否则返回false</returns>
	bool is_loaded() noexcept;

	/// <summary>
	/// 清空标签内存.
	/// </summary>
	/// <returns></returns>
	void free_lables() noexcept;

	/// <summary>
	/// 识别整个车牌.
	/// </summary>
	/// <param name="roi">车牌位图.</param>
	/// <param name="results">返回识别出来的车牌的索引 -> car_ids.</param>
	/// <returns></returns>
	void analyse(cv::Mat* roi, int results[7]) noexcept;

	/// <summary>
	/// 识别车牌的某个字符.
	/// </summary>
	/// <param name="buffer">该车牌字符的位图.</param>
	/// <returns>返回识别出来的车牌字符的索引 -> car_ids.</returns>
	int get_per_car_id(cv::Mat buffer) noexcept;

	/// <summary>
	/// 将cv::Mat格式转化为image格式.
	/// </summary>
	/// <param name="mat">cv::Mat格式的车牌位图.</param>
	/// <returns>image格式的车牌位图</returns>
	image to_image(cv::Mat mat) noexcept;
};
