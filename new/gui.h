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

	//d3d9设备
	IDirect3D9* m_IDirect3D9;
	IDirect3DDevice9* m_IDirect3DDevice9;
	D3DPRESENT_PARAMETERS m_D3DPRESENT_PARAMETERS;
	IDirect3DTexture9* m_IDirect3DTexture9;

	//视频处理
	video m_video;

private:
	//窗口过程
	static r_hresult __stdcall window_proc(r_hwnd _hwnd, r_uint msg, r_wparam wpa, r_lparam lpa) noexcept;

	//初始化d3d9设备
	void initialize_d3d9() noexcept;

	//初始化imgui界面库
	void initialize_imgui() noexcept;

	//渲染处理
	void render_handle() noexcept;

	//重置d3d9设备
	void device_reset() noexcept;

	//清理d3d9设备
	void clear_d3d9() noexcept;

	//清理imgui界面库
	void clear_imgui() noexcept;

	//转化为utf8编码
	char* to_utf8(const char* text, char* buffer, int size) noexcept;

	//更新图片到纹理
	void update_texture(struct frame_handle* data) noexcept;

	//视频显示窗口
	void imgui_display_video() noexcept;

	//选择视频文件菜单
	void imgui_select_video(bool update = true) noexcept;

	//覆盖窗口 视频控制选项
	void imgui_video_control_overlay(ImVec2 pos, float width) noexcept;

	//区域管理功能
	void imgui_region_manager();

	//人流量统计窗口
	void imgui_people_statistics();

	//窗口主菜单
	void imgui_window_meun() noexcept;

	//模型窗口
	void imgui_model_window() noexcept;

	//功能窗口
	void imgui_features_window() noexcept;

	//窗口
	void imgui_win_window() noexcept;

	//语言窗口
	void imgui_language_window() noexcept;

public:
	gui() noexcept;

	//创建窗口和显示
	void create_and_show(const char* name = "darknet_imgui") noexcept;

	//窗口消息处理
	void msg_handle() noexcept;

	//通知窗口大小发生改变
	void notice_reset(int w, int h) noexcept;

	//设置英文显示
	void set_english_display(bool enable) noexcept;
};
