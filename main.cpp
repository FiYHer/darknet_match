#include "main.h"

global_set g_global_set;

//int main(int argc, char* argv[])
//{
//	//�����Կ�����
//	cuda_set_device(cuda_get_device());
//	CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
//
//	std::map<std::string, int> class_names;
//	class_names.insert({ "car",1 });
//	class_names.insert({ "truck" ,1 });
//	class_names.insert({ "person" ,2 });
//	class_names.insert({ "motorbike",3 });
//	class_names.insert({ "bicycle" ,4 });
//	class_names.insert({ "traffic light",5 });
//	class_names.insert({ "dog",6 });
//	class_names.insert({ "bus",7 });
//
//	std::vector<std::string> buffer
//	{
//		"E:\\PascalVoc\\VOC2012\\JPEGImages"
//	};
//
//	for (auto& it : buffer) picture_to_label(it.c_str(), class_names);
//	printf("������!------------------------------------------------------");
//	getchar();
//	return 0;
//}

int main(int argc, char* argv[])//cmd���ڲ�������
////int _stdcall WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
{
	//���ù����Կ��͹���ģʽ
	cuda_set_device(cuda_get_device());
	CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));

	register_window_struct();//ע�ᴰ����
	create_window();//��������
	initialize_d3d9();//��ʼ��d3d�豸
	initialize_imgui();//��ʼ��imgui�����
	window_message_handle();//������Ϣ����
	clear_imgui_set();//����imgui�����
	clear_d3d9_set();//����d3d�豸
	return 0;
}

//����Ĭ�ϵ�imgui����ⴰ�ڹ���
extern LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
LRESULT _stdcall window_process(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam)
{
	//imgui����⴦���˵���Ϣ���ǲ�������
	if (ImGui_ImplWin32_WndProcHandler(hwnd, msg, wparam, lparam)) return true;

	//���ڴ�С�ı�ᵼ���豸��ʧ���������ǻ�ȡ���ڴ��´�С��Ҫ����d3d�豸����
	if (msg == WM_SIZE && wparam != SIZE_MINIMIZED)
	{
		g_global_set.d3dpresent.BackBufferWidth = LOWORD(lparam);
		g_global_set.d3dpresent.BackBufferHeight = HIWORD(lparam);
		reset_d3d9_device();
		return true;
	}

	//�˳�������Ϣ
	if (msg == WM_DESTROY) PostQuitMessage(0);

	//����û�˴����Ĭ�ϴ���
	return DefWindowProcA(hwnd, msg, wparam, lparam);
}

void register_window_struct()
{
	//��ʼ��һ���������
	srand((unsigned int)time(0));

	//���ô��ڵı���
	sprintf(g_global_set.window_class_name, "���ڼ�����Ӿ��Ľ�ͨ��������Ӧ��__%d", rand());

	//��ʼ��������
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

	//ע�ᴰ����
	check_serious_error(RegisterClassExA(&window_class), "ע�ᴰ����ʧ��");
}

void create_window()
{
	//���Դ�������
	g_global_set.window_hwnd = CreateWindowExA(NULL, g_global_set.window_class_name, g_global_set.window_class_name,
		WS_OVERLAPPEDWINDOW, 100, 100, 1000, 600, NULL, NULL, GetModuleHandleA(NULL), NULL);
	check_serious_error(g_global_set.window_hwnd, "��������ʧ��");

	//�������õĴ��ڽ�����ʾ
	ShowWindow(g_global_set.window_hwnd, SW_SHOWMAXIMIZED);
	UpdateWindow(g_global_set.window_hwnd);
}

void initialize_d3d9()
{
	//��ȡd3dָ��
	g_global_set.direct3d9 = Direct3DCreate9(D3D_SDK_VERSION);
	check_serious_error(g_global_set.direct3d9, "��ʼ��d3d9ʧ��");

	//��ȡd3d9�豸ָ��
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
	IMGUI_CHECKVERSION();//���imgui�����汾
	ImGui::CreateContext();//����һ������������
	ImGuiIO& io = ImGui::GetIO(); (void)io;

	//�������κ������ļ�
	io.IniFilename = NULL;
	io.LogFilename = NULL;

	//���ý�����Ϊ��ɫ
	ImGui::StyleColorsLight();

	//�󶨲���
	ImGui_ImplWin32_Init(g_global_set.window_hwnd);
	ImGui_ImplDX9_Init(g_global_set.direct3ddevice9);

	//����΢���ź����壬Ϊ��֧�����ĵ���ʾ
	const char* font_path = "msyh.ttc";
	check_serious_error(_access(font_path, 0) != -1, "΢�����������ļ�ȱʧ");
	io.Fonts->AddFontFromFileTTF(font_path, 20.0f, NULL, io.Fonts->GetGlyphRangesChineseFull());
}

void window_message_handle()
{
	//���ϻ�ȡ������Ϣ��û�д�����Ϣ�ͽ��н���Ļ��Ʋ���
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
	//û��ͼƬ����
	if (!g_global_set.picture_set.data) return;

	//d3d9�豸Ϊ��
	if (!g_global_set.direct3ddevice9) return;

	//��ȡ��� �߶� ͨ����
	int width = g_global_set.picture_set.w;
	int height = g_global_set.picture_set.h;
	int channel = g_global_set.picture_set.c;

	//����ͼƬ����
	HRESULT result = g_global_set.direct3ddevice9->CreateTexture(width, height, 1, D3DUSAGE_DYNAMIC, D3DFMT_X8R8G8B8, D3DPOOL_DEFAULT, &g_global_set.direct3dtexture9, NULL);
	if (result == S_OK)
	{
		//���������ڴ�
		D3DLOCKED_RECT lock_rect;
		g_global_set.direct3dtexture9->LockRect(0, &lock_rect, NULL, 0);

		//��ͼƬ���ݿ����������ڴ���
		for (int i = 0; i < height; i++)
			memcpy((unsigned char *)lock_rect.pBits + lock_rect.Pitch * i, g_global_set.picture_set.data + (width * channel) * i, (width * channel));

		//�ͷ������ڴ�
		g_global_set.direct3dtexture9->UnlockRect(0);
	}
}

void clear_picture_texture()
{
	//����������ͷ�
	if (g_global_set.direct3dtexture9) g_global_set.direct3dtexture9->Release();
	g_global_set.direct3dtexture9 = nullptr;
}

void imgui_show_handle()
{
	//��ͼƬ���ݿ���������
	picture_to_texture();

	//��Ⱦ���׼��
	ImGui_ImplDX9_NewFrame();
	ImGui_ImplWin32_NewFrame();
	ImGui::NewFrame();

	//��ʼ��Ⱦһϵ�д���
	ImGui::ShowDemoWindow();
	imgui_show_manager();
	imgui_object_detect_window();
	imgui_car_id_identify_window();
	imgui_test_picture_window();
	imgui_test_video_window();
	imgui_load_region_window();

	//����Ⱦ�����Ľ�����Ƶ���������ش���
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

	//�ͷ�ͼƬ�����ڴ�
	clear_picture_texture();
}

void imgui_show_manager()
{
	//���ô��ڵĴ�С��λ��
	ImGui::SetNextWindowSize(ImVec2(500, 300), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowPos(ImVec2(0, 0));
	ImGui::Begin(u8"���ܽ�ͨϵͳ");

	//��ȡ�Կ�����
	static int gpu_count = cuda_get_device();

	//��ȡ�Կ������Ϣ
	static cudaDeviceProp* gpu_info = get_gpu_infomation(gpu_count);

	//��ʾ�Կ������Ϣ
	for (int i = 0; i < gpu_count; i++)
		ImGui::BulletText(u8"GPU�ͺ�\t %s", gpu_info[i].name);

	//��ȡϵͳ����
	static int os_type = get_os_type();
	if (os_type == -1) ImGui::BulletText(u8"ϵͳ����\t δ֪");
	else ImGui::BulletText(u8"ϵͳ����\t Windows %d", os_type);

	//��ʾCPU������
	static int cpu_kernel = get_cpu_kernel();
	ImGui::BulletText(u8"CPU����\t %d", cpu_kernel);

	//��ʾ�ڴ�����
	static int phy_memory = get_physical_memory();
	ImGui::BulletText(u8"�����ڴ�\t %d", phy_memory);

	ImGui::Separator();
	if (ImGui::Button(u8"������ģ������")) g_global_set.imgui_show_set.show_object_detect_window = true;
	ImGui::SameLine();
	if (ImGui::Button(u8"����ʶ��ģ������")) g_global_set.imgui_show_set.show_car_id_identify_window = true;
	if (ImGui::Button(u8"��ͨͼƬ������")) g_global_set.imgui_show_set.show_test_picture_window = true;
	ImGui::SameLine();
	if (ImGui::Button(u8"��ͨ��Ƶ������")) g_global_set.imgui_show_set.show_test_video_window = true;

	ImGui::End();
}

void imgui_object_detect_window()
{
	if (!g_global_set.imgui_show_set.show_object_detect_window) return;
	ImGui::SetNextWindowSize(ImVec2(500, 300), ImGuiCond_FirstUseEver);
	ImGui::Begin(u8"������", &g_global_set.imgui_show_set.show_object_detect_window);

	//��ȡ���������С
	const ImVec2 size = ImGui::GetContentRegionAvail();

	static char names_path[default_char_size] = "object_detect.names";
	ImGui::InputText(u8"*names�ļ�·��", (char*)string_to_utf8((const char*)names_path).c_str(), default_char_size);

	static char cfg_path[default_char_size] = "object_detect.cfg";
	ImGui::InputText(u8"*.cfg�ļ�·��", (char*)string_to_utf8((const char*)cfg_path).c_str(), default_char_size);

	static char weights_path[default_char_size] = "object_detect.weights";
	ImGui::InputText(u8"*.weights�ļ�·��", (char*)string_to_utf8((const char*)weights_path).c_str(), default_char_size);

	if (ImGui::Button(u8"ѡ��*.names�ļ�", ImVec2(size.x / 3, 0.0f))) select_type_file("names flie\0*.names\0\0", names_path);
	ImGui::SameLine();
	if (ImGui::Button(u8"ѡ��*.cfg�ļ�", ImVec2(size.x / 3, 0.0f))) select_type_file("cfg file\0*.cfg\0\0", cfg_path);
	ImGui::SameLine();
	if (ImGui::Button(u8"ѡ��*.weights�ļ�", ImVec2(size.x / 3, 0.0f))) select_type_file("weights file\0*.weights\0\0", weights_path);
	ImGui::Separator();

	ImGui::BulletText(u8"��ʼ��״̬ : "); ImGui::SameLine();
	if(g_global_set.object_detect_net_set.initizlie) ImGui::TextColored(ImVec4(255, 0, 0, 255), u8"[ �� ]");
	else ImGui::TextColored(ImVec4(0, 0, 255, 255), u8"[ �� ]");

	if (ImGui::Button(u8"��ʼ������������", ImVec2(size.x / 2, 0.0f))) initialize_object_detect_net(names_path, cfg_path, weights_path);
	ImGui::SameLine();
	if (ImGui::Button(u8"�ͷ�����������", ImVec2(size.x / 2, 0.0f))) clear_object_detect_net();

	ImGui::End();
}

void imgui_car_id_identify_window()
{
	if (!g_global_set.imgui_show_set.show_car_id_identify_window) return;
	ImGui::SetNextWindowSize(ImVec2(500, 300), ImGuiCond_FirstUseEver);
	ImGui::Begin(u8"����ʶ��", &g_global_set.imgui_show_set.show_car_id_identify_window);

	//��ȡ���������С
	const ImVec2 size = ImGui::GetContentRegionAvail();

	static char names_path[default_char_size] = "car_id.names";
	ImGui::InputText(u8"*names�ļ�·��", (char*)string_to_utf8((const char*)names_path).c_str(), default_char_size);

	static char cfg_path[default_char_size] = "car_id.cfg";
	ImGui::InputText(u8"*.cfg�ļ�·��", (char*)string_to_utf8((const char*)cfg_path).c_str(), default_char_size);

	static char weights_path[default_char_size] = "car_id.weights";
	ImGui::InputText(u8"*.weights�ļ�·��", (char*)string_to_utf8((const char*)weights_path).c_str(), default_char_size);

	if (ImGui::Button(u8"ѡ��*.names�ļ�", ImVec2(size.x / 3, 0.0f))) select_type_file("names flie\0*.names\0\0", names_path);
	ImGui::SameLine();
	if (ImGui::Button(u8"ѡ��*.cfg�ļ�", ImVec2(size.x / 3, 0.0f))) select_type_file("cfg file\0*.cfg\0\0", cfg_path);
	ImGui::SameLine();
	if (ImGui::Button(u8"ѡ��*.weights�ļ�", ImVec2(size.x / 3, 0.0f))) select_type_file("weights file\0*.weights\0\0", weights_path);
	ImGui::Separator();

	ImGui::BulletText(u8"��ʼ��״̬ : ");
	ImGui::SameLine();
	if (g_global_set.car_id_identify_net.initizlie) ImGui::TextColored(ImVec4(255, 0, 0, 255), u8"[ �� ]");
	else ImGui::TextColored(ImVec4(0, 0, 255, 255), u8"[ �� ]");

	if (ImGui::Button(u8"��ʼ������ʶ������", ImVec2(size.x / 2, 0.0f))) initialize_car_id_identify_net(names_path, cfg_path, weights_path);
	ImGui::SameLine();
	if (ImGui::Button(u8"�ͷų���ʶ������", ImVec2(size.x / 2, 0.0f))) clear_car_id_identify_net();

	ImGui::End();
}

void imgui_test_picture_window()
{
	if (!g_global_set.imgui_show_set.show_test_picture_window) return;

	ImGui::SetNextWindowSize(ImVec2(300, 300), ImGuiCond_FirstUseEver);
	ImGui::Begin(u8"���ܽ�ͨϵͳ  -  ����ͼƬ����", &g_global_set.imgui_show_set.show_test_picture_window);

	static set_detect_info detect_info;

	//ͼƬ������
	if (ImGui::CollapsingHeader(u8"����"))
	{
		ImGui::InputFloat(u8"thresh", &detect_info.thresh, 0.01f, 1.0f, "%.3f");
		ImGui::InputFloat(u8"hier_thresh", &detect_info.hier_thresh, 0.01f, 1.0f, "%.3f");
		ImGui::InputFloat(u8"nms", &detect_info.nms, 0.01f, 1.0f, "%.3f");
	}

	//��ɫ����
	if (ImGui::CollapsingHeader(u8"��ɫ"))
	{
		ImGui::ColorEdit3(u8"������ɫ", g_global_set.color_set.box_rgb);
		ImGui::ColorEdit3(u8"������ɫ", g_global_set.color_set.font_rgb);
		ImGui::InputFloat(u8"�������", &g_global_set.color_set.thickness, 0.01f, 1.0f, "%.3f");
	}

	//��ȡ���������С
	const ImVec2 size = ImGui::GetContentRegionAvail();

	static char target_picture[default_char_size] = "match.jpg";
	ImGui::InputText(u8"����ͼƬ", (char*)string_to_utf8(target_picture).c_str(), default_char_size);
	if (ImGui::Button(u8"ѡ��ͼƬ", ImVec2(size.x / 2, 0.0f))) select_type_file("image file\0*.jpg;*.bmp;*.png\0\0", target_picture, default_char_size);
	ImGui::SameLine();

	if (ImGui::Button(u8"ִ�м��", ImVec2(size.x / 2, 0.0f)))
	{
		if (g_global_set.object_detect_net_set.initizlie) analyse_picture(target_picture, detect_info);
		else show_window_tip("����û�г�ʼ��");
	}

	ImGui::BulletText(u8"�������ʱ : %.2lfms ", detect_info.detect_time);
	ImGui::BulletText(u8"����ʶ���ʱ : %.2lfms", detect_info.identify_time);

	ImGui::End();
}

void imgui_test_video_window()
{
	if (!g_global_set.imgui_show_set.show_test_video_window) return;

	ImGui::SetNextWindowSize(ImVec2(500, 700), ImGuiCond_FirstUseEver);
	ImGui::Begin(u8"���ܽ�ͨϵͳ  -  ������Ƶ����", &g_global_set.imgui_show_set.show_test_video_window);

	//��Ƶ����
	static video_control control_info;
	if (ImGui::CollapsingHeader(u8"�ӳ�"))
	{
		ImGui::InputInt(u8"��ʾ�ӳ�", &control_info.show_delay);
		ImGui::InputInt(u8"��ȡ�ӳ�", &control_info.read_delay);
		ImGui::InputInt(u8"����ӳ�", &control_info.detect_delay);
		ImGui::InputInt(u8"�����ӳ�", &control_info.scene_delay);
	}

	if (ImGui::CollapsingHeader(u8"���"))
	{
		ImGui::InputFloat(u8"thresh", &g_global_set.video_detect_set.thresh, 0.1f, 1.0f, "%.2f");
		ImGui::InputFloat(u8"hier_thresh", &g_global_set.video_detect_set.hier_thresh, 0.1f, 1.0f, "%.2f");
		ImGui::InputFloat(u8"nms", &g_global_set.video_detect_set.nms, 0.1f, 1.0f, "%.2f");
		ImGui::InputInt(u8"����߳�", &control_info.detect_count);
	}

	if (ImGui::CollapsingHeader(u8"����"))
	{
		ImGui::Checkbox(u8"ͳ��������", &g_global_set.secne_set.human_traffic);
		ImGui::Checkbox(u8"ͳ�Ƴ�����", &g_global_set.secne_set.car_traffic);
		ImGui::Checkbox(u8"ռ�ù�������", &g_global_set.secne_set.occupy_bus_lane);
		ImGui::Checkbox(u8"�����", &g_global_set.secne_set.rush_red_light);
		ImGui::Checkbox(u8"����������ʻ", &g_global_set.secne_set.not_guided);
		ImGui::Checkbox(u8"�����߲���������", &g_global_set.secne_set.not_zebra_cross);
	}

	if (ImGui::CollapsingHeader(u8"����"))
	{
		ImGui::ColorEdit3(u8"������ɫ", g_global_set.color_set.box_rgb);
		ImGui::ColorEdit3(u8"������ɫ", g_global_set.color_set.font_rgb);
		ImGui::InputFloat(u8"�������", &g_global_set.color_set.thickness, 0.01f, 1.0f, "%.3f");
		ImGui::InputInt2(u8"��Ƶ��С", control_info.video_size);
	}

	ImGui::RadioButton(u8"��Ƶģʽ", &control_info.use_camera, 0);
	ImGui::SameLine();
	ImGui::RadioButton(u8"����ͷģʽ", &control_info.use_camera, 1);

	if (control_info.use_camera) ImGui::InputInt(u8"����ͷ����", &control_info.camera_index);
	else
	{
		ImGui::InputText(u8"��Ƶ·��", (char*)string_to_utf8(control_info.video_path).c_str(), default_char_size);
		if (ImGui::Button(u8"ѡ����Ƶ", ImVec2(-FLT_MIN, 0.0f))) select_type_file("video file\0*.mp4;*.avi;*.flv;*.ts\0\0", control_info.video_path);
	}

	//��ȡ����ʾ�����С
	const ImVec2 region_size = ImGui::GetContentRegionAvail();

	if (ImGui::Button(u8"�������", ImVec2(region_size.x / 3, 0.0f))) g_global_set.imgui_show_set.show_set_load_region_window = true;
	ImGui::SameLine();
	if (ImGui::Button(u8"��ʼ���", ImVec2(region_size.x / 3, 0.0f)))
	{
		if (!control_info.leave) show_window_tip("����ֹͣ��һ����Ƶ�ļ��");
		else if (g_global_set.object_detect_net_set.initizlie) _beginthreadex(NULL, 0, analyse_video, &control_info, 0, NULL);
		else show_window_tip("����û�г�ʼ��");
	}
	ImGui::SameLine();
	if (ImGui::Button(u8"ֹͣ���", ImVec2(region_size.x / 3, 0.0f))) control_info.leave = true;
	
	if (g_global_set.secne_set.human_traffic)
	{
		ImGui::Separator();

		//��ֹ0
		int tick_size = g_global_set.secne_set.human_num.size();
		if (!tick_size) tick_size = 1;

		//�����ڴ�
		float *temp = new float[tick_size];
		memset(temp, 0, tick_size * sizeof(float));

		int total_val = 0;
		static int max_val = 0;
		for (int i = 0; i < g_global_set.secne_set.human_num.size(); i++)
		{
			temp[i] = g_global_set.secne_set.human_num[i];
			total_val += temp[i];//��������
			if (max_val < temp[i]) max_val = temp[i];//��������
		}

		//����ƽ��������
		g_global_set.secne_set.human_avg = total_val / tick_size;

		ImGui::BulletText(u8"�������� : %d ", g_global_set.secne_set.human_count); ImGui::SameLine();
		ImGui::BulletText(u8"��ǰ������ : %d ", g_global_set.secne_set.human_current); ImGui::SameLine();
		ImGui::BulletText(u8"ƽ�������� : %d ", g_global_set.secne_set.human_avg);

		//��������ͼ
		ImGui::PlotHistogram(u8"ÿ��������", temp, tick_size, 0, NULL, 0.0f, (float)max_val, ImVec2(region_size.x, 50.0f));
		
		//�ͷ��ڴ�
		delete[] temp;

		//����
		if (ImGui::Button(u8"����������", ImVec2(-FLT_MIN, 0.0f)))
		{
			g_global_set.secne_set.human_count = 0;
			g_global_set.secne_set.human_current = 0;
			g_global_set.secne_set.human_avg = 0;
			g_global_set.secne_set.human_minute = 0;
			g_global_set.secne_set.human_num.clear();

			max_val = 0;
		}
	}

	if (g_global_set.secne_set.car_traffic)
	{
		ImGui::Separator();
		
		//��ȡ����
		int tick_size = g_global_set.secne_set.car_num.size();
		if (!tick_size) tick_size = 1;
		
		//�����ڴ�
		float *temp = new float[tick_size];
		memset(temp, 0, tick_size * sizeof(float));

		int tatol_val = 0;
		static int max_val = 0;
		for (int i = 0; i < g_global_set.secne_set.car_num.size(); i++)
		{
			temp[i] = g_global_set.secne_set.car_num[i];
			if (max_val < temp[i]) max_val = temp[i];
			tatol_val += temp[i];
		}

		//����ƽ��ֵ
		g_global_set.secne_set.car_avg = tatol_val / tick_size;

		ImGui::BulletText(u8"�ܳ����� : %d ", g_global_set.secne_set.car_count); ImGui::SameLine();
		ImGui::BulletText(u8"��ǰ������ : %d ", g_global_set.secne_set.car_current); ImGui::SameLine();
		ImGui::BulletText(u8"ƽ�������� : %d ", g_global_set.secne_set.car_avg);

		//��������ͼ
		ImGui::PlotHistogram(u8"ÿ��������", temp, tick_size, 0, NULL, 0.0f, (float)max_val, ImVec2(region_size.x, 50.0f));

		//�ͷ��ڴ�
		delete temp;

		if (ImGui::Button(u8"���ó�����", ImVec2(-FLT_MIN, 0.0f)))
		{
			g_global_set.secne_set.car_count = 0;
			g_global_set.secne_set.car_current = 0;
			g_global_set.secne_set.car_avg = 0;
			g_global_set.secne_set.car_minute = 0;
			g_global_set.secne_set.car_num.clear();

			max_val = 0;
		}
	}

	if (g_global_set.secne_set.occupy_bus_lane && g_global_set.secne_set.occupy_bus_list.size())
	{
		//����ֻ��ʾ5������Ŀ��Ա�����ļ�����
		int size = g_global_set.secne_set.occupy_bus_list.size();
		if (size > 5)
			g_global_set.secne_set.occupy_bus_list.erase(g_global_set.secne_set.occupy_bus_list.begin(),
				g_global_set.secne_set.occupy_bus_list.begin() + (size - 5));

		//��ʾ
		for (auto& it : g_global_set.secne_set.occupy_bus_list)
			ImGui::BulletText(u8"ռ�ù������� - %d��%d��%d��%dʱ%d��%d��",
				it.times[0], it.times[1], it.times[2], it.times[3], it.times[4], it.times[5]);
		ImGui::Separator();
	}

	ImGui::End();
}

void imgui_load_region_window()
{
	if (!g_global_set.imgui_show_set.show_set_load_region_window) return;

	//���ô��ڵô�С��λ��
	ImGui::SetNextWindowSize(ImVec2{ 600,600 }, ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowPos(ImVec2{ 0,0 });
	ImGui::Begin(u8"������", &g_global_set.imgui_show_set.show_set_load_region_window, ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoMove);

	//��ȡ���ƾ��
	ImDrawList* draw_list = ImGui::GetWindowDrawList();

	//��ȡ��С��λ��
	const ImVec2 window_size = { ImGui::GetWindowWidth(),ImGui::GetWindowHeight() };
	const ImVec2 current_pos = { ImGui::GetIO().MousePos.x,ImGui::GetIO().MousePos.y };

	//������ɫ
	static ImVec4 colf = ImVec4(0.0f, 1.0f, 0.0f, 1.0f);
	const ImU32 col = ImColor(colf);

	//������Ϣ
	static region_mask region_info;

	//���� �� ��ʼλ��
	static int set_step = 0;
	static ImVec2 start_pos{ -1,-1 };

	//��� ���¿�ʼ
	auto clear_start_pos = []()
	{
		set_step = 0;
		start_pos = { -1,-1 };
	};

	//�Ȼ���ͼƬ
	if (g_global_set.direct3dtexture9)
		ImGui::Image(g_global_set.direct3dtexture9, ImGui::GetContentRegionAvail());

	//�������
	if (ImGui::BeginMenuBar())
	{
		static char video_path[default_char_size];
		if (ImGui::BeginMenu(u8"�������"))
		{
			if (ImGui::Button(u8"��ȡһ֡��Ƶͼ��") && select_type_file("video file\0*.mp4;*.avi;*.flv\0\0", video_path, default_char_size))
			{
				read_video_frame(video_path);
				clear_start_pos();
			}
			if (ImGui::Button(u8"��ǰ����ߵ�ͨ��"))
			{
				colf.w = colf.y = 1.0f; colf.x = colf.z = 0;
				region_info.type = region_zebra_cross;
				clear_start_pos();
			}
			if (ImGui::Button(u8"��ǹ�����ר�õ�"))
			{
				colf.w = colf.z = 1.0f; colf.y = colf.x = 0;
				region_info.type = region_bus_lane;
				clear_start_pos();
			}
			if (ImGui::Button(u8"�����·��ͣ��λ"))
			{
				colf.w = colf.x = 1.0f; colf.y = colf.z = 0;
				region_info.type = region_street_parking;
				clear_start_pos();
			}
			ImGui::EndMenu();
		}
		if (ImGui::BeginMenu(u8"ʹ��˵��"))
		{
			ImGui::BulletText(u8"��ȡһ֡��Ƶ");
			ImGui::BulletText(u8"ѡ�񷽿�������");
			ImGui::BulletText(u8"�������������ÿ�ʼλ��");
			ImGui::BulletText(u8"�ٴε������������ý���λ��");
			ImGui::BulletText(u8"��������Ҽ�ɾ����һ��������");
			ImGui::EndMenu();
		}
		ImGui::EndMenuBar();
	}

	//����������
	if (ImGui::IsMouseClicked(0))
	{
		//��������λ�ò���ͼ
		if (current_pos.x <= 0 || current_pos.y - 50.0f <= 0 || current_pos.x >= window_size.x || current_pos.y >= window_size.y) {}
		else if (set_step == 0)//���ÿ�ʼλ��
		{
			start_pos = current_pos;
			set_step++;
		}
		else if (set_step == 1)//��������б�
		{
			//����̫С
			if (abs(current_pos.x - start_pos.x) < 10.0f && abs(current_pos.y - start_pos.y) < 10.0f) {}
			else
			{
				region_info.window_size = { window_size.x,window_size.y - 50.0f };//���洰�ڴ�С
				region_info.pos = start_pos;//���濪ʼλ��
				region_info.size = current_pos;//�������
				region_info.rect_color = colf;//������ɫ
				g_global_set.mask_list.push_back(region_info);
				clear_start_pos();
			}
		}
	}

	//����Ҽ�����
	if (ImGui::IsMouseClicked(1))
	{
		if (!set_step && g_global_set.mask_list.size()) g_global_set.mask_list.pop_back();
		else clear_start_pos();
	}

	//��ʾ�����еķ���
	for (auto& it : g_global_set.mask_list) draw_list->AddRect(it.pos, it.size, ImColor(it.rect_color), 0.0f, 0, 1.0f);

	//���Ƶ�ǰ�ķ���
	if (start_pos.x != -1 && start_pos.y != -1) draw_list->AddRect(start_pos, current_pos, col, 0.0f, 0, 1.0f);

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
	g_global_set.direct3d9 = nullptr;
	g_global_set.direct3ddevice9 = nullptr;
}

