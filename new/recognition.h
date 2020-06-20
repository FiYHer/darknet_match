#pragma once

#include "darknet.h"
#include "parser.h"
#include "opencv2/opencv.hpp"

#include "help.h"

#include <fstream>
#include <string>
#include <vector>

/// <summary>
/// ���ƺ�
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
"��",
"��",
"��",
"��",
"��",
"��",
"��",
"��",
"��",
"��",
"��",
"��",
"��",
"��",
"��",
"³",
"ԥ",
"��",
"��",
"��",
"��",
"��",
"��",
"��",
"��",
"��",
"��",
"��",
"��",
"��",
"��",
"��",
"ѧ" };

/// <summary>
/// ����ʶ����
/// </summary>
class object_recognition
{
private:
	//�Ƿ��ʼ��ģ��
	bool m_initialize_model;

	//ģ���ļ�·��
	char m_path[max_string_len];

	//ʶ������
	network m_net;

	//��ǩ����
	char** m_labels_name;

	//��ǩ����
	int m_labels_count;

public:
	object_recognition() noexcept;
	~object_recognition() noexcept;

	/// <summary>
	/// ����ģ���ļ�·��.
	/// </summary>
	/// <param name="path">ģ���ļ�·��.</param>
	/// <returns>���óɹ�����true�����򷵻�false</returns>
	bool set_model_path(const char* path) noexcept;

	/// <summary>
	/// ����ģ��.
	/// </summary>
	/// <returns>���سɹ�����true�����򷵻�false</returns>
	bool load_model() noexcept;

	/// <summary>
	/// ж��ģ��.
	/// </summary>
	/// <returns>ж�سɹ�����true�����򷵻�false</returns>
	bool unload_model() noexcept;

	/// <summary>
	/// �ж��Ƿ����ģ��.
	/// </summary>
	/// <returns>ģ�ͼ����˷���true�����򷵻�false</returns>
	bool is_loaded() noexcept;

	/// <summary>
	/// ��ձ�ǩ�ڴ�.
	/// </summary>
	/// <returns></returns>
	void free_lables() noexcept;

	/// <summary>
	/// ʶ����������.
	/// </summary>
	/// <param name="roi">����λͼ.</param>
	/// <param name="results">����ʶ������ĳ��Ƶ����� -> car_ids.</param>
	/// <returns></returns>
	void analyse(cv::Mat* roi, int results[7]) noexcept;

	/// <summary>
	/// ʶ���Ƶ�ĳ���ַ�.
	/// </summary>
	/// <param name="buffer">�ó����ַ���λͼ.</param>
	/// <returns>����ʶ������ĳ����ַ������� -> car_ids.</returns>
	int get_per_car_id(cv::Mat buffer) noexcept;

	/// <summary>
	/// ��cv::Mat��ʽת��Ϊimage��ʽ.
	/// </summary>
	/// <param name="mat">cv::Mat��ʽ�ĳ���λͼ.</param>
	/// <returns>image��ʽ�ĳ���λͼ</returns>
	image to_image(cv::Mat mat) noexcept;
};
