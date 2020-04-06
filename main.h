#pragma once
#include "common.h"

//窗口过程
LRESULT _stdcall window_process(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam);

//注册窗口结构
void register_window_struct();

//创建窗口
void create_window();

//初始化d3d9设备
void initialize_d3d9();

//初始化imgui
void initialize_imgui();

//窗口消息循环处理
void window_message_handle();

//加载图片数据到纹理
void picture_to_texture();

//释放图片纹理数据
void clear_picture_texture();

//imgui界面显示处理
void imgui_show_handle();

//imgui界面管理器
void imgui_show_manager();

//文件设置窗口
void imgui_file_set_window();

//测试图片窗口
void imgui_test_picture_window();

//测试视频窗口
void imgui_test_video_window();

//显示马路区域设置窗口
void imgui_load_region_window();

//重置d3d9设备
void reset_d3d9_device();

//清空imgui设置
void clear_imgui_set();

//清空d3d9设备
void clear_d3d9_set();

