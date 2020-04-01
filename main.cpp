#include "main.h"

#include "imgui/imgui.h"
#include "imgui/imgui_impl_win32.h"
#include "imgui/imgui_impl_dx9.h"

global_set g_global_set;

int main(int argc, char* argv[])
{
	register_window_struct();
	create_window();
	initialize_d3d9();
	initialize_imgui();
	window_message_handle();
	clear_imgui_set();
	clear_d3d9_set();
	return 0;
}

extern LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
LRESULT _stdcall window_process(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam)
{
	if (ImGui_ImplWin32_WndProcHandler(hwnd, msg, wparam, lparam))
		return true;

	if (msg == WM_SIZE && wparam != SIZE_MINIMIZED)
	{
		g_global_set.d3dpresent.BackBufferWidth = LOWORD(lparam);
		g_global_set.d3dpresent.BackBufferHeight = HIWORD(lparam);
		reset_d3d9_device();
		return true;
	}

	if (msg == WM_DESTROY) PostQuitMessage(0);

	return DefWindowProcA(hwnd, msg, wparam, lparam);
}

void register_window_struct()
{
	sprintf(g_global_set.window_class_name, "�й������_��ͨ����ϵͳ_���ֵ��ӿƼ���ѧ����У��_���Գ���_%d", rand());

	WNDCLASSEXA window_class{
		sizeof(WNDCLASSEXA),
		CS_CLASSDC,
		window_process,
		0L,
		0L,
		GetModuleHandleA(NULL),
		NULL,
		NULL,
		NULL,
		NULL,
		g_global_set.window_class_name,
	};
	check_serious_error(RegisterClassExA(&window_class),"ע�ᴰ����ʧ��");
}

void create_window()
{
	g_global_set.window_hwnd = CreateWindowExA(NULL, g_global_set.window_class_name, g_global_set.window_class_name,
		WS_OVERLAPPEDWINDOW, 100, 100, 1000, 600, NULL, NULL, GetModuleHandleA(NULL), NULL);
	check_serious_error(g_global_set.window_hwnd, "��������ʧ��");

	ShowWindow(g_global_set.window_hwnd, SW_SHOW);
	UpdateWindow(g_global_set.window_hwnd);
}

void initialize_d3d9()
{
	g_global_set.direct3d9 = Direct3DCreate9(D3D_SDK_VERSION);
	check_serious_error(g_global_set.direct3d9, "��ʼ��d3d9ʧ��");

	ZeroMemory(&g_global_set.d3dpresent, sizeof(g_global_set.d3dpresent));
	g_global_set.d3dpresent.Windowed = TRUE;
	g_global_set.d3dpresent.SwapEffect = D3DSWAPEFFECT_DISCARD;
	g_global_set.d3dpresent.BackBufferFormat = D3DFMT_UNKNOWN;
	g_global_set.d3dpresent.EnableAutoDepthStencil = TRUE;
	g_global_set.d3dpresent.AutoDepthStencilFormat = D3DFMT_D16;
	g_global_set.d3dpresent.PresentationInterval = D3DPRESENT_INTERVAL_ONE;
	check_serious_error(g_global_set.direct3d9->CreateDevice(D3DADAPTER_DEFAULT,
		D3DDEVTYPE_HAL, g_global_set.window_hwnd, D3DCREATE_HARDWARE_VERTEXPROCESSING, &g_global_set.d3dpresent, &g_global_set.direct3ddevice9) == S_OK, "��ʼ��d3d9deviceʧ��");
}

void initialize_imgui()
{
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	io.IniFilename = NULL;
	io.LogFilename = NULL;

	ImGui::StyleColorsLight();
	ImGui_ImplWin32_Init(g_global_set.window_hwnd);
	ImGui_ImplDX9_Init(g_global_set.direct3ddevice9);

	const char* font_path = "msyh.ttc";
	check_serious_error(_access(font_path, 0) != -1, "�����ļ�ȱʧ");
	io.Fonts->AddFontFromFileTTF(font_path, 20.0f, NULL, io.Fonts->GetGlyphRangesChineseFull());
}

void window_message_handle()
{
	MSG msg{ 0 };
	while (msg.message != WM_QUIT)
	{
		if (!PeekMessageA(&msg, NULL, 0U, 0U, PM_REMOVE)) imgui_show_handle();
		else
		{
			TranslateMessage(&msg);
			DispatchMessageA(&msg);
		}
	}
}

void picture_to_texture()
{
	//ֻ�ڲ���ͼƬ������ʾ
	if (!g_global_set.imgui_show_set.show_test_picture_window) return;

	//û��ͼƬ����
	if (!g_global_set.picture_data) return;

	//d3d9�豸Ϊ��
	if (!g_global_set.direct3ddevice9) return;

	int width = g_global_set.width;
	int height = g_global_set.height;
	int channel = g_global_set.channel;
	HRESULT result = g_global_set.direct3ddevice9->CreateTexture(width, height, 1, D3DUSAGE_DYNAMIC, D3DFMT_X8R8G8B8, D3DPOOL_DEFAULT, &g_global_set.direct3dtexture9, NULL);
	check_serious_error(result == S_OK, "����ͼƬ���ݵ�����ʧ��");
	D3DLOCKED_RECT lock_rect;
	g_global_set.direct3dtexture9->LockRect(0, &lock_rect, NULL, 0);
	for (int i = 0; i < g_global_set.height; i++) 
		memcpy((unsigned char *)lock_rect.pBits + lock_rect.Pitch * i, g_global_set.picture_data + (width * channel) * i, (width * channel));
	g_global_set.direct3dtexture9->UnlockRect(0);
}

void clear_picture_texture()
{
	if (g_global_set.direct3dtexture9) g_global_set.direct3dtexture9->Release();
	g_global_set.direct3dtexture9 = nullptr;
}

void imgui_show_handle()
{
	picture_to_texture();

	ImGui_ImplDX9_NewFrame();
	ImGui_ImplWin32_NewFrame();
	ImGui::NewFrame();

	//ImGui::ShowDemoWindow();
	imgui_show_manager();
	imgui_file_set_window();
	imgui_test_picture_window();
	imgui_test_video_window();
	imgui_test_camera_window();

	ImGui::EndFrame();
	static D3DCOLOR background_color = D3DCOLOR_RGBA(200, 200, 200, 0);
	g_global_set.direct3ddevice9->SetRenderState(D3DRS_ZENABLE, false);
	g_global_set.direct3ddevice9->SetRenderState(D3DRS_ALPHABLENDENABLE, false);
	g_global_set.direct3ddevice9->SetRenderState(D3DRS_SCISSORTESTENABLE, false);
	g_global_set.direct3ddevice9->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, background_color, 1.0f, 0);
	if (g_global_set.direct3ddevice9->BeginScene() >= 0)
	{
		ImGui::Render();
		ImGui_ImplDX9_RenderDrawData(ImGui::GetDrawData());
		g_global_set.direct3ddevice9->EndScene();
	}
	HRESULT result = g_global_set.direct3ddevice9->Present(NULL, NULL, NULL, NULL);
	if (result == D3DERR_DEVICELOST 
		&& g_global_set.direct3ddevice9->TestCooperativeLevel() == D3DERR_DEVICENOTRESET) reset_d3d9_device();

	clear_picture_texture();
}

void imgui_show_manager()
{
	ImGui::SetNextWindowSize(ImVec2(500, 300), ImGuiCond_FirstUseEver);

	ImGui::Begin(u8"���ܽ�ͨϵͳ");

	if (ImGui::Button(u8"�ļ����ô���")) g_global_set.imgui_show_set.show_file_set_window = true;

	if (ImGui::Button(u8"����ͼƬϵͳ����")) g_global_set.imgui_show_set.show_test_picture_window = true;
	ImGui::SameLine();
	if (ImGui::Button(u8"������Ƶϵͳ����")) g_global_set.imgui_show_set.show_test_video_window = true;
	ImGui::SameLine();
	if (ImGui::Button(u8"��������ͷϵͳ����")) g_global_set.imgui_show_set.show_test_picture_window = true;

	ImGui::End();
}

void imgui_file_set_window()
{
	if (!g_global_set.imgui_show_set.show_file_set_window) return;

	ImGui::SetNextWindowSize(ImVec2(500,350), ImGuiCond_FirstUseEver);
	ImGui::Begin(u8"���ܽ�ͨϵͳ  -  �ļ����ô���", &g_global_set.imgui_show_set.show_file_set_window);

	ImGui::TextColored(ImVec4(255, 0, 0, 255), u8"û����������벻Ҫ����Ķ��ļ�·��!!!");

	static char data_path[default_char_size] = "match.data";
	ImGui::InputText(u8"*.data�ļ�·��", data_path, default_char_size);

	static char names_path[default_char_size] = "match.names";
	ImGui::InputText(u8"*names�ļ�·��", names_path, default_char_size);

	static char cfg_path[default_char_size] = "match.cfg";
	ImGui::InputText(u8"*.cfg�ļ�·��", cfg_path, default_char_size);

	static char weights_path[default_char_size] = "match.weights";
	ImGui::InputText(u8"*.weights�ļ�·��", weights_path, default_char_size);

	static int classes_number = 20;
	ImGui::Text(u8"������� : %d", classes_number);

	if (ImGui::Button(u8"ѡ��*.data�ļ�"))
	{
		if (select_type_file("data flie\0*.data\0\0", data_path))
		{
			list* data_list = read_data_cfg(data_path);
			classes_number = option_find_int(data_list, "classes", 0);
			char* temp = option_find_str(data_list, "names", "default.names");
			if (temp) strncpy(names_path, temp, strlen(temp));

			free_list_contents_kvp(data_list);
			free_list(data_list);
		}
	}
	ImGui::SameLine();
	if (ImGui::Button(u8"ѡ��*.cfg�ļ�")) select_type_file("cfg file\0*.cfg\0\0", cfg_path);
	ImGui::SameLine();
	if (ImGui::Button(u8"ѡ��*.weights�ļ�")) select_type_file("weights\0*.weights\0\0", weights_path);
	
	ImGui::Separator();
	static bool load_net = false;
	if (!load_net) ImGui::TextColored(ImVec4(0, 0, 255, 255), u8"û�г�ʼ������ �����ʼ���Ῠ��<=20�� ��ȴ�....");
	else ImGui::TextColored(ImVec4(0, 255, 0, 255), u8"��ʼ������ɹ�....");

	if (ImGui::Button(u8"��ʼ������"))
	{
		strncpy(g_global_set.net_set.data_path, data_path, strlen(data_path));
		strncpy(g_global_set.net_set.names_path, names_path, strlen(names_path));
		strncpy(g_global_set.net_set.cfg_path, cfg_path, strlen(cfg_path));
		strncpy(g_global_set.net_set.weights_path, weights_path, strlen(weights_path));
		g_global_set.net_set.classes = classes_number;
		if (initialize_net()) load_net = true;
		else load_net = false;
	}
	ImGui::SameLine();
	if (ImGui::Button(u8"��������"))
	{
		clear_net();
		load_net = false;
	}

	ImGui::End();
}

void imgui_test_picture_window()
{
	if (!g_global_set.imgui_show_set.show_test_picture_window) return;

	ImGui::SetNextWindowSize(ImVec2(800, 500), ImGuiCond_FirstUseEver);
	ImGui::Begin(u8"���ܽ�ͨϵͳ  -  ����ͼƬ����",&g_global_set.imgui_show_set.show_test_picture_window);

	static char target_picture[default_char_size] = "match.jpg";
	ImGui::InputText(u8"����ͼƬ", target_picture, default_char_size);
	ImGui::SameLine();
	if (ImGui::Button(u8"ѡ��ͼƬ")) select_type_file("image file\0*.jpg;0*.bmp;0*.png\0\0", target_picture, default_char_size);
	ImGui::SameLine();

	//��ȡ�Ķ�����Ϣ
	static std::vector<object_info> object;

	if (ImGui::Button(u8"ִ�м��"))
	{
		if (g_global_set.net_set.initizlie) analyse_picture(target_picture, object);
		else show_window_tip("����û�г�ʼ��");
	}

	//��ʾ��������
	if (object.size())
	{
		char temp[default_char_size];
		sprintf(temp, "��⵽�������� : %d\t ����ʱ : %lf����", object.size(), g_global_set.detection_time);
		ImGui::BulletText(string_to_utf8(temp).c_str());

		for (auto& it : object)
		{
			sprintf(temp, "Object : %s\t Confidence : %f\t left : %d  top : %d  right : %d  down : %d \n",
				g_global_set.net_set.classes_name[it.class_index], it.confidence, it.left, it.top, it.right, it.down);
			ImGui::BulletText(temp);
		}
	}

	//����ͼ����������ʾͼƬ
	if (g_global_set.direct3dtexture9)
		ImGui::Image(g_global_set.direct3dtexture9, ImGui::GetContentRegionAvail());

	ImGui::End();
}

void imgui_test_video_window()
{
	if (!g_global_set.imgui_show_set.show_test_video_window) return;

	ImGui::SetNextWindowSize(ImVec2(800, 600), ImGuiCond_FirstUseEver);
	ImGui::Begin(u8"���ܽ�ͨϵͳ  -  ������Ƶ����",&g_global_set.imgui_show_set.show_test_video_window);

	ImGui::SliderInt(u8"��ʾ��Ƶ�ӳ�", &g_global_set.show_video_delay, 1, 100);
	ImGui::SliderInt(u8"��ȡ��Ƶ�ӳ�", &g_global_set.read_video_delay, 1, 100);
	ImGui::SliderInt(u8"�����Ƶ�ӳ�", &g_global_set.detect_video_delay, 1, 100);

	static char video_path[default_char_size] = "match.mp4";
	ImGui::InputText(u8"��Ƶ·��", video_path, default_char_size);
	ImGui::SameLine();
	if (ImGui::Button(u8"ѡ����Ƶ")) select_type_file("video file\0*.mp4;0*.avi\0\0", video_path);
	ImGui::SameLine();
	if (ImGui::Button(u8"��ʼ���"))
	{
		if(g_global_set.net_set.initizlie) analyse_video(video_path);
		else show_window_tip("����û�г�ʼ��");
	}


	ImGui::End();
}

void imgui_test_camera_window()
{
	if (!g_global_set.imgui_show_set.show_test_camera_window) return;

	ImGui::Begin(u8"���ܽ�ͨϵͳ  -  ��������ͷ����", &g_global_set.imgui_show_set.show_test_camera_window);

	ImGui::End();
}

void reset_d3d9_device()
{
	if (g_global_set.direct3ddevice9)
	{
		ImGui_ImplDX9_InvalidateDeviceObjects();
		g_global_set.direct3ddevice9->Reset(&g_global_set.d3dpresent);
		ImGui_ImplDX9_CreateDeviceObjects();
	}
}

void clear_imgui_set()
{
	ImGui_ImplDX9_Shutdown();
	ImGui_ImplWin32_Shutdown();
	ImGui::DestroyContext();
}

void clear_d3d9_set()
{
	if (g_global_set.direct3d9) g_global_set.direct3d9->Release();
	if (g_global_set.direct3ddevice9) g_global_set.direct3ddevice9->Release();
}

