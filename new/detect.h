#pragma once

#include "help.h"
#include "darknet.h"
#include "parser.h"

/// <summary>
/// ��������
/// </summary>
class object_detect
{
private:
	//ģ��·���ļ�
	char* m_path;

	//��ǩ����
	char** m_classes_name;
	int m_classes_count;

	//��ǩ��ɫ
	float** m_classes_color;

	//����
	network m_net;

	//�Ƿ����
	bool m_loaded;

	//��������ֵ
	float m_thresh;
	float m_hier_thresh;
	float m_nms;

public:
	object_detect();
	~object_detect();

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

public:
	/// <summary>
	/// ����ģ���ļ�·��.
	/// </summary>
	/// <param name="path">ģ���ļ�·��.</param>
	/// <returns>���óɹ�����true�����򷵻�false</returns>
	bool set_model_path(const char* path) noexcept;

	/// <summary>
	/// ��ȡ����ģ��.
	/// </summary>
	/// <returns>����ģ��ָ��</returns>
	network* get_network() noexcept;

	/// <summary>
	/// ��ȡ��ǩ����.
	/// </summary>
	/// <returns>�����������</returns>
	int get_classes_count() const noexcept;

	/// <summary>
	/// ��ȡ��ǩ�ַ���.
	/// </summary>
	/// <returns>��������ַ���ָ��</returns>
	char** get_classes_name() const noexcept;

	/// <summary>
	/// ��ȡģ���Ƿ����.
	/// </summary>
	/// <returns>����ģ�ͼ��ص�״̬��ʶ</returns>
	bool get_model_loader() const noexcept;

	/// <summary>
	/// ��ȡ�����ֵ���.
	/// </summary>
	/// <returns>���������ֵ</returns>
	float get_thresh() const noexcept;
	float get_hier_thresh() const noexcept;
	float get_nms() const noexcept;

	/// <summary>
	/// ��ȡ�����ɫ.
	/// </summary>
	/// <returns>���������ɫָ��</returns>
	float** get_classes_color() const noexcept;

	/// <summary>
	/// �ͷ���������ڴ�.
	/// </summary>
	/// <returns></returns>
	void free_classes_name() noexcept;

	/// <summary>
	/// �ͷ������ɫ�ڴ�.
	/// </summary>
	/// <returns></returns>
	void free_classes_color() noexcept;
};
