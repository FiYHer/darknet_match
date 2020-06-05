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
	//�������ڴ�С�ı�ʱ֪ͨ�豸��ʧ
	static gui* m_static_this;

private:
	//���ھ��
	r_hwnd m_hwnd;

	//Ӣ����ʾ
	bool m_is_english;

	//���������ʾ
	bool m_region_manager;

	//������ͳ����ʾ
	bool m_people_statistics;

	//������ͳ����ʾ
	bool m_car_statistics;

	//d3d9�豸
	IDirect3D9* m_IDirect3D9;
	IDirect3DDevice9* m_IDirect3DDevice9;
	D3DPRESENT_PARAMETERS m_D3DPRESENT_PARAMETERS;
	IDirect3DTexture9* m_IDirect3DTexture9;

	//ͼƬ����
	std::vector<IDirect3DTexture9*> m_textures_list;

	//��Ƶ����
	video m_video;

private:
	/// <summary>
	/// ���ڹ��̺���
	/// </summary>
	/// <param name="_hwnd">���ھ��.</param>
	/// <param name="msg">������Ϣ.</param>
	/// <param name="wpa">��Ϣ���Ӳ���.</param>
	/// <param name="lpa">��Ϣ���Ӳ���.</param>
	/// <returns></returns>
	static r_hresult __stdcall window_proc(r_hwnd _hwnd, r_uint msg, r_wparam wpa, r_lparam lpa) noexcept;

	/// <summary>
	/// ��ʼ��d3d9�豸.
	/// </summary>
	/// <returns></returns>
	void initialize_d3d9() noexcept;

	/// <summary>
	/// ��ʼ��imgui�����.
	/// </summary>
	/// <returns></returns>
	void initialize_imgui() noexcept;

	/// <summary>
	/// ��Ⱦ����.
	/// </summary>
	/// <returns></returns>
	void render_handle() noexcept;

	/// <summary>
	/// ����d3d9�豸.
	/// </summary>
	/// <returns></returns>
	void device_reset() noexcept;

	/// <summary>
	/// ����d3d9�豸.
	/// </summary>
	/// <returns></returns>
	void clear_d3d9() noexcept;

	/// <summary>
	/// ����imgui�����.
	/// </summary>
	/// <returns></returns>
	void clear_imgui() noexcept;

	/// <summary>
	/// ת��Ϊutf8����.
	/// </summary>
	/// <param name="text">��Ҫ����ΪUTF8���ַ���.</param>
	/// <param name="buffer">��Ž��������.</param>
	/// <param name="size">��������С.</param>
	/// <returns>���ػ�����ָ��</returns>
	char* to_utf8(const char* text, char* buffer, int size) noexcept;

	/// <summary>
	/// ����ͼƬ������.
	/// </summary>
	/// <param name="data">��Ƶ֡����.</param>
	/// <returns></returns>
	void update_texture(struct frame_handle* data) noexcept;

	/// <summary>
	/// ��ȡͼƬ����
	/// </summary>
	/// <param name="frame">ͼƬ����ָ��</param>
	/// <returns></returns>
	IDirect3DTexture9* get_image_texture(cv::Mat* frame) noexcept;

	/// <summary>
	/// �ͷ�ȫ����ͼ��λͼ.
	/// </summary>
	/// <returns></returns>
	void release_image_texture() noexcept;

	/// <summary>
	/// ��Ƶ��ʾ����.
	/// </summary>
	/// <returns></returns>
	void imgui_display_video() noexcept;

	/// <summary>
	/// ѡ����Ƶ�ļ��˵�.
	/// </summary>
	/// <param name="update">�Ƿ������Ƶ�����Ƶ·��</param>
	/// <returns></returns>
	void imgui_select_video(bool update = true) noexcept;

	/// <summary>
	/// ���Ǵ��� ��Ƶ����ѡ��.
	/// </summary>
	/// <param name="pos">��ʾ��λ��.</param>
	/// <param name="width">���.</param>
	/// <returns></returns>
	void imgui_video_control_overlay(ImVec2 pos, float width) noexcept;

	/// <summary>
	/// ���������.
	/// </summary>
	/// <returns></returns>
	void imgui_region_manager() noexcept;

	/// <summary>
	/// ������ͳ�ƴ���.
	/// </summary>
	/// <returns></returns>
	void imgui_people_statistics() noexcept;

	/// <summary>
	/// ������ͳ�ƴ���.
	/// </summary>
	/// <returns></returns>
	void imgui_car_statistics() noexcept;

	/// <summary>
	/// �������˵�.
	/// </summary>
	/// <returns></returns>
	void imgui_window_meun() noexcept;

	/// <summary>
	/// ģ�ʹ���.
	/// </summary>
	/// <returns></returns>
	void imgui_model_window() noexcept;

	/// <summary>
	/// ���ܴ���.
	/// </summary>
	/// <returns></returns>
	void imgui_features_window() noexcept;

	/// <summary>
	/// ����.
	/// </summary>
	/// <returns></returns>
	void imgui_win_window() noexcept;

	/// <summary>
	/// ���Դ���.
	/// </summary>
	/// <returns></returns>
	void imgui_language_window() noexcept;

public:
	gui() noexcept;
	~gui() noexcept;

	/// <summary>
	/// �������ں���ʾ.
	/// </summary>
	/// <param name="name">The name.</param>
	/// <returns></returns>
	void create_and_show(const char* name = "darknet_imgui") noexcept;

	/// <summary>
	/// ������Ϣ����.
	/// </summary>
	/// <returns></returns>
	void msg_handle() noexcept;

	/// <summary>
	/// ֪ͨ���ڴ�С�����ı�.
	/// </summary>
	/// <param name="w">���ڿ��.</param>
	/// <param name="h">���ڸ߶�.</param>
	/// <returns></returns>
	void notice_reset(int w, int h) noexcept;

	/// <summary>
	/// ����Ӣ����ʾ.
	/// </summary>
	/// <param name="enable">�Ƿ�����Ӣ����ʾ.</param>
	/// <returns></returns>
	void set_english_display(bool enable) noexcept;
};
