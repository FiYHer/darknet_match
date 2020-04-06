#pragma once
#include "common.h"

//���ڹ���
LRESULT _stdcall window_process(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam);

//ע�ᴰ�ڽṹ
void register_window_struct();

//��������
void create_window();

//��ʼ��d3d9�豸
void initialize_d3d9();

//��ʼ��imgui
void initialize_imgui();

//������Ϣѭ������
void window_message_handle();

//����ͼƬ���ݵ�����
void picture_to_texture();

//�ͷ�ͼƬ��������
void clear_picture_texture();

//imgui������ʾ����
void imgui_show_handle();

//imgui���������
void imgui_show_manager();

//�ļ����ô���
void imgui_file_set_window();

//����ͼƬ����
void imgui_test_picture_window();

//������Ƶ����
void imgui_test_video_window();

//��ʾ��·�������ô���
void imgui_load_region_window();

//����d3d9�豸
void reset_d3d9_device();

//���imgui����
void clear_imgui_set();

//���d3d9�豸
void clear_d3d9_set();

