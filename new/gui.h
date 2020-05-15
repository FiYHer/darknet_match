#pragma once

#include <d3d9.h>
#pragma comment(lib,"d3d9.lib")

#include "imgui/imgui.h"
#include "imgui/imgui_impl_dx9.h"
#include "imgui/imgui_impl_win32.h"

#include "help.h"
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

	//d3d9�豸
	IDirect3D9* m_IDirect3D9;
	IDirect3DDevice9* m_IDirect3DDevice9;
	D3DPRESENT_PARAMETERS m_D3DPRESENT_PARAMETERS;
	IDirect3DTexture9* m_IDirect3DTexture9;

	//��Ƶ����
	video m_video;

private:
	//���ڹ���
	static r_hresult __stdcall window_proc(r_hwnd _hwnd, r_uint msg, r_wparam wpa, r_lparam lpa) noexcept;

	//��ʼ��d3d9�豸
	void initialize_d3d9() noexcept;

	//��ʼ��imgui�����
	void initialize_imgui() noexcept;

	//��Ⱦ����
	void render_handle() noexcept;

	//����d3d9�豸
	void device_reset() noexcept;

	//����d3d9�豸
	void clear_d3d9() noexcept;

	//����imgui�����
	void clear_imgui() noexcept;

	//ת��Ϊutf8����
	char* to_utf8(const char* text, char* buffer, int size) noexcept;

	//����ͼƬ������
	void update_texture(struct frame_handle* data) noexcept;

	//��Ƶ��ʾ����
	void imgui_display_video() noexcept;

	//���Ǵ��� ��Ƶ����ѡ��
	void imgui_video_control_overlay() noexcept;

	//�������˵�
	void imgui_window_meun() noexcept;

	//�ļ�����
	void imgui_file_window() noexcept;

	//ģ�ʹ���
	void imgui_model_window() noexcept;

	//����
	void imgui_win_window() noexcept;

	//ģʽ����
	void imgui_mode_window() noexcept;

	//���Դ���
	void imgui_language_window() noexcept;




public:
	gui() noexcept;

	//�������ں���ʾ
	void create_and_show(const char* name = "darknet_imgui") noexcept;

	//������Ϣ����
	void msg_handle() noexcept;

	//֪ͨ���ڴ�С�����ı�
	void notice_reset(int w, int h) noexcept;

	//����Ӣ����ʾ
	void set_english_display(bool enable) noexcept;


};
