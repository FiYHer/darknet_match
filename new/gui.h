#pragma once

#include <d3d9.h>
#pragma comment(lib,"d3d9.lib")

#include "imgui/imgui.h"
#include "imgui/imgui_impl_dx9.h"
#include "imgui/imgui_impl_win32.h"

#include "video.h"

class gui
{
private:
	//辅助窗口大小改变时通知设备丢失
	static gui* m_static_this;

private:
	//窗口句柄
	r_hwnd m_hwnd;

	//英文显示
	bool m_is_english;

	//区域管理显示
	bool m_region_manager;

	//人流量统计显示
	bool m_people_statistics;

	//车流量统计显示
	bool m_car_statistics;

	//d3d9设备
	IDirect3D9* m_IDirect3D9;
	IDirect3DDevice9* m_IDirect3DDevice9;
	D3DPRESENT_PARAMETERS m_D3DPRESENT_PARAMETERS;
	IDirect3DTexture9* m_IDirect3DTexture9;

	//图片纹理
	std::vector<IDirect3DTexture9*> m_textures_list;

	//视频处理
	video m_video;

private:
	/// <summary>
	/// 窗口过程函数
	/// </summary>
	/// <param name="_hwnd">窗口句柄.</param>
	/// <param name="msg">窗口消息.</param>
	/// <param name="wpa">消息附加参数.</param>
	/// <param name="lpa">消息附加参数.</param>
	/// <returns></returns>
	static r_hresult __stdcall window_proc(r_hwnd _hwnd, r_uint msg, r_wparam wpa, r_lparam lpa) noexcept;

	/// <summary>
	/// 初始化d3d9设备.
	/// </summary>
	/// <returns></returns>
	void initialize_d3d9() noexcept;

	/// <summary>
	/// 初始化imgui界面库.
	/// </summary>
	/// <returns></returns>
	void initialize_imgui() noexcept;

	/// <summary>
	/// 渲染处理.
	/// </summary>
	/// <returns></returns>
	void render_handle() noexcept;

	/// <summary>
	/// 重置d3d9设备.
	/// </summary>
	/// <returns></returns>
	void device_reset() noexcept;

	/// <summary>
	/// 清理d3d9设备.
	/// </summary>
	/// <returns></returns>
	void clear_d3d9() noexcept;

	/// <summary>
	/// 清理imgui界面库.
	/// </summary>
	/// <returns></returns>
	void clear_imgui() noexcept;

	/// <summary>
	/// 转化为utf8编码.
	/// </summary>
	/// <param name="text">需要编码为UTF8的字符串.</param>
	/// <param name="buffer">存放结果缓冲区.</param>
	/// <param name="size">缓冲区大小.</param>
	/// <returns>返回缓冲区指针</returns>
	char* to_utf8(const char* text, char* buffer, int size) noexcept;

	/// <summary>
	/// 更新图片到纹理.
	/// </summary>
	/// <param name="data">视频帧数据.</param>
	/// <returns></returns>
	void update_texture(struct frame_handle* data) noexcept;

	/// <summary>
	/// 获取图片纹理
	/// </summary>
	/// <param name="frame">图片数据指针</param>
	/// <returns></returns>
	IDirect3DTexture9* get_image_texture(cv::Mat* frame) noexcept;

	/// <summary>
	/// 释放全部的图像位图.
	/// </summary>
	/// <returns></returns>
	void release_image_texture() noexcept;

	/// <summary>
	/// 视频显示窗口.
	/// </summary>
	/// <returns></returns>
	void imgui_display_video() noexcept;

	/// <summary>
	/// 选择视频文件菜单.
	/// </summary>
	/// <param name="update">是否更新视频类的视频路径</param>
	/// <returns></returns>
	void imgui_select_video(bool update = true) noexcept;

	/// <summary>
	/// 覆盖窗口 视频控制选项.
	/// </summary>
	/// <param name="pos">显示的位置.</param>
	/// <param name="width">宽度.</param>
	/// <returns></returns>
	void imgui_video_control_overlay(ImVec2 pos, float width) noexcept;

	/// <summary>
	/// 区域管理功能.
	/// </summary>
	/// <returns></returns>
	void imgui_region_manager() noexcept;

	/// <summary>
	/// 人流量统计窗口.
	/// </summary>
	/// <returns></returns>
	void imgui_people_statistics() noexcept;

	/// <summary>
	/// 车流量统计窗口.
	/// </summary>
	/// <returns></returns>
	void imgui_car_statistics() noexcept;

	/// <summary>
	/// 窗口主菜单.
	/// </summary>
	/// <returns></returns>
	void imgui_window_meun() noexcept;

	/// <summary>
	/// 模型窗口.
	/// </summary>
	/// <returns></returns>
	void imgui_model_window() noexcept;

	/// <summary>
	/// 功能窗口.
	/// </summary>
	/// <returns></returns>
	void imgui_features_window() noexcept;

	/// <summary>
	/// 窗口.
	/// </summary>
	/// <returns></returns>
	void imgui_win_window() noexcept;

	/// <summary>
	/// 语言窗口.
	/// </summary>
	/// <returns></returns>
	void imgui_language_window() noexcept;

public:
	gui() noexcept;
	~gui() noexcept;

	/// <summary>
	/// 创建窗口和显示.
	/// </summary>
	/// <param name="name">The name.</param>
	/// <returns></returns>
	void create_and_show(const char* name = "darknet_imgui") noexcept;

	/// <summary>
	/// 窗口消息处理.
	/// </summary>
	/// <returns></returns>
	void msg_handle() noexcept;

	/// <summary>
	/// 通知窗口大小发生改变.
	/// </summary>
	/// <param name="w">窗口宽度.</param>
	/// <param name="h">窗口高度.</param>
	/// <returns></returns>
	void notice_reset(int w, int h) noexcept;

	/// <summary>
	/// 设置英文显示.
	/// </summary>
	/// <param name="enable">是否启用英文显示.</param>
	/// <returns></returns>
	void set_english_display(bool enable) noexcept;
};
