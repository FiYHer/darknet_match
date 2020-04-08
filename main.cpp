#include "main.h"

global_set g_global_set;

//int main(int argc, char* argv[])
//{
//	//设置显示工作
//	int gpu_index;
//	cudaGetDeviceCount(&gpu_index);
//	cuda_set_device(0);
//	CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
//
//	std::map<std::string, int> class_names;
//	class_names.insert({ "car",1 });
//	class_names.insert({ "train" ,1 });
//	class_names.insert({ "truck" ,1 });
//	class_names.insert({ "person" ,2});
//	class_names.insert({ "motorbike",3 });
//	class_names.insert({ "bicycle" ,4 });
//	class_names.insert({ "traffic light",5 });
//	class_names.insert({ "dog",6 });
//	class_names.insert({ "bus",7 });
//
//	std::vector<std::string> buffer
//	{
//		"H:\\CarPicture\\Coco2017",
//		"H:\\CarPicture\\Insight-MVT_Annotation_Train\\MVI_20011",
//		"H:\\CarPicture\\Insight-MVT_Annotation_Train\\MVI_20012",
//		"H:\\CarPicture\\Insight-MVT_Annotation_Train\\MVI_20032",
//		"H:\\CarPicture\\Insight-MVT_Annotation_Train\\MVI_20033",
//		"H:\\CarPicture\\Insight-MVT_Annotation_Train\\MVI_20034",
//		"H:\\CarPicture\\Insight-MVT_Annotation_Train\\MVI_20035",
//		"H:\\CarPicture\\Insight-MVT_Annotation_Train\\MVI_20051",
//		"H:\\CarPicture\\Insight-MVT_Annotation_Train\\MVI_20052",
//		"H:\\CarPicture\\Insight-MVT_Annotation_Train\\MVI_20061",
//		"H:\\CarPicture\\Insight-MVT_Annotation_Train\\MVI_20062",
//		"H:\\CarPicture\\Insight-MVT_Annotation_Train\\MVI_20063",
//		"H:\\CarPicture\\Insight-MVT_Annotation_Train\\MVI_20064",
//		"H:\\CarPicture\\Insight-MVT_Annotation_Train\\MVI_20065",
//		"H:\\CarPicture\\Insight-MVT_Annotation_Train\\MVI_39761",
//		"H:\\CarPicture\\run_load",
//		"H:\\CarPicture\\run_load1",
//		"H:\\CarPicture\\run_load2",
//		"H:\\CarPicture\\run_load3",
//		"H:\\CarPicture\\run_load4",
//		"H:\\CarPicture\\run_load5"
//	};
//	for(auto& it : buffer)
//		picture_to_label(it.c_str(), class_names);
//	printf("标记完成!------------------------------------------------------");
//	getchar();
//	return 0;
//}

//int main(int argc,char* argv[])
int _stdcall WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
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

	//窗口大小改变导致设备丢失
	if (msg == WM_SIZE && wparam != SIZE_MINIMIZED)
	{
		g_global_set.d3dpresent.BackBufferWidth = LOWORD(lparam);
		g_global_set.d3dpresent.BackBufferHeight = HIWORD(lparam);
		reset_d3d9_device();
		return true;
	}

	//退出程序
	if (msg == WM_DESTROY) PostQuitMessage(0);

	return DefWindowProcA(hwnd, msg, wparam, lparam);
}

void register_window_struct()
{
	srand((unsigned int)time(0));
	sprintf(g_global_set.window_class_name, "中国软件杯_交通智能系统_桂林电子科技大学北海校区_测试程序_%d", rand());

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
	check_serious_error(RegisterClassExA(&window_class),"注册窗口类失败");
}

void create_window()
{
	g_global_set.window_hwnd = CreateWindowExA(NULL, g_global_set.window_class_name, g_global_set.window_class_name,
		WS_OVERLAPPEDWINDOW, 100, 100, 1000, 600, NULL, NULL, GetModuleHandleA(NULL), NULL);
	check_serious_error(g_global_set.window_hwnd, "创建窗口失败");

	ShowWindow(g_global_set.window_hwnd, SW_SHOWMAXIMIZED);
	UpdateWindow(g_global_set.window_hwnd);
}

void initialize_d3d9()
{
	g_global_set.direct3d9 = Direct3DCreate9(D3D_SDK_VERSION);
	check_serious_error(g_global_set.direct3d9, "初始化d3d9失败");

	ZeroMemory(&g_global_set.d3dpresent, sizeof(g_global_set.d3dpresent));
	g_global_set.d3dpresent.Windowed = TRUE;
	g_global_set.d3dpresent.SwapEffect = D3DSWAPEFFECT_DISCARD;
	g_global_set.d3dpresent.BackBufferFormat = D3DFMT_UNKNOWN;
	g_global_set.d3dpresent.EnableAutoDepthStencil = TRUE;
	g_global_set.d3dpresent.AutoDepthStencilFormat = D3DFMT_D16;
	g_global_set.d3dpresent.PresentationInterval = D3DPRESENT_INTERVAL_ONE;
	check_serious_error(g_global_set.direct3d9->CreateDevice(D3DADAPTER_DEFAULT,
		D3DDEVTYPE_HAL, g_global_set.window_hwnd, D3DCREATE_HARDWARE_VERTEXPROCESSING, &g_global_set.d3dpresent, &g_global_set.direct3ddevice9) == S_OK, "初始化d3d9device失败");
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
	check_serious_error(_access(font_path, 0) != -1, "微软字体字体文件缺失");
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
	//没有图片数据
	if (!g_global_set.picture_set.data) return;

	//d3d9设备为空
	if (!g_global_set.direct3ddevice9) return;

	//获取宽度 高度 通道数
	int width = g_global_set.picture_set.w;
	int height = g_global_set.picture_set.h;
	int channel = g_global_set.picture_set.c;

	//创建图片纹理
	HRESULT result = g_global_set.direct3ddevice9->CreateTexture(width, height, 1, D3DUSAGE_DYNAMIC, D3DFMT_X8R8G8B8, D3DPOOL_DEFAULT, &g_global_set.direct3dtexture9, NULL);
	if (result == S_OK)
	{
		//锁定纹理内存
		D3DLOCKED_RECT lock_rect;
		g_global_set.direct3dtexture9->LockRect(0, &lock_rect, NULL, 0);

		//将图片数据拷贝到纹理内存中
		for (int i = 0; i < height; i++)
			memcpy((unsigned char *)lock_rect.pBits + lock_rect.Pitch * i, g_global_set.picture_set.data + (width * channel) * i, (width * channel));

		//释放锁定内存
		g_global_set.direct3dtexture9->UnlockRect(0);
	}
}

void clear_picture_texture()
{
	if (g_global_set.direct3dtexture9) g_global_set.direct3dtexture9->Release();
	g_global_set.direct3dtexture9 = nullptr;
}

void imgui_show_handle()
{
	//将图片数据拷贝到纹理
	picture_to_texture();

	//渲染相关准备
	ImGui_ImplDX9_NewFrame();
	ImGui_ImplWin32_NewFrame();
	ImGui::NewFrame();

	//开始渲染一系列窗口
	ImGui::ShowDemoWindow();
	imgui_show_manager();
	imgui_file_set_window();
	imgui_test_picture_window();
	imgui_test_video_window();
	imgui_load_region_window();

	//将渲染出来的界面绘制到窗口上相关代码
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

	//释放图片纹理内存
	clear_picture_texture();
}

void imgui_show_manager()
{
	ImGui::SetNextWindowSize(ImVec2(500, 300), ImGuiCond_FirstUseEver);

	ImGui::Begin(u8"智能交通系统");

	//获取显卡数量
	static int gpu_count = get_gpu_count();

	//获取显卡相关信息
	static cudaDeviceProp* gpu_info = get_gpu_infomation(gpu_count);

	//显示显卡相关信息
	for (int i = 0; i < gpu_count; i++)
		ImGui::BulletText(u8"GPU型号  [%s]", gpu_info[i].name);

	//获取系统类型
	static int os_type = get_os_type();
	if(os_type == -1) ImGui::BulletText(u8"系统类型  [未知]");
	else ImGui::BulletText(u8"系统类型  [Windows %d]", os_type);

	//显示CPU核心数
	static int cpu_kernel = get_cpu_kernel();
	ImGui::BulletText(u8"CPU核心  [%d]", cpu_kernel);

	//显示内存总数
	static int phy_memory = get_physical_memory();
	ImGui::BulletText(u8"物理内存  [%dg]", phy_memory);

	ImGui::Separator();
	if (ImGui::Button(u8"文件配置窗口")) g_global_set.imgui_show_set.show_file_set_window = true;
	ImGui::SameLine();
	if (ImGui::Button(u8"测试图片系统窗口")) g_global_set.imgui_show_set.show_test_picture_window = true;
	ImGui::SameLine();
	if (ImGui::Button(u8"测试视频系统窗口")) g_global_set.imgui_show_set.show_test_video_window = true;

	ImGui::End();
}

void imgui_file_set_window()
{
	if (!g_global_set.imgui_show_set.show_file_set_window) return;

	ImGui::SetNextWindowSize(ImVec2(500,350), ImGuiCond_FirstUseEver);
	ImGui::Begin(u8"智能交通系统  -  文件设置窗口", &g_global_set.imgui_show_set.show_file_set_window);

	static char names_path[default_char_size] = "match.names";
	ImGui::InputText(u8"*names文件路径", (char*)string_to_utf8((const char*)names_path).c_str(), default_char_size);

	static char cfg_path[default_char_size] = "match.cfg";
	ImGui::InputText(u8"*.cfg文件路径", (char*)string_to_utf8((const char*)cfg_path).c_str(), default_char_size);

	static char weights_path[default_char_size] = "match.weights";
	ImGui::InputText(u8"*.weights文件路径", (char*)string_to_utf8((const char*)weights_path).c_str(), default_char_size);

	const ImVec2 size = ImGui::GetContentRegionAvail();

	if (ImGui::Button(u8"选择*.names文件", ImVec2(size.x / 3, 0.0f))) select_type_file("names flie\0*.names\0\0", names_path);
	ImGui::SameLine();
	if (ImGui::Button(u8"选择*.cfg文件", ImVec2(size.x / 3, 0.0f))) select_type_file("cfg file\0*.cfg\0\0", cfg_path);
	ImGui::SameLine();
	if (ImGui::Button(u8"选择*.weights文件", ImVec2(size.x / 3, 0.0f))) select_type_file("weights file\0*.weights\0\0", weights_path);
	
	ImGui::Separator();
	static bool load_net = false;
	if (!load_net) ImGui::TextColored(ImVec4(0, 0, 255, 255), u8"网络初始化会卡顿20秒 请等待....");
	else ImGui::TextColored(ImVec4(255, 0, 0, 255), u8"初始化网络成功....");

	if (ImGui::Button(u8"初始化网络", ImVec2(size.x / 2, 0.0f)) && !load_net) load_net = initialize_net(names_path, cfg_path, weights_path);
	ImGui::SameLine();
	if (ImGui::Button(u8"清理网络", ImVec2(size.x / 2, 0.0f)))
	{
		clear_net();
		load_net = false;
	}

	ImGui::End();
}

void imgui_test_picture_window()
{
	if (!g_global_set.imgui_show_set.show_test_picture_window) return;

	ImGui::SetNextWindowSize(ImVec2(300, 300), ImGuiCond_FirstUseEver);
	ImGui::Begin(u8"智能交通系统  -  测试图片窗口",&g_global_set.imgui_show_set.show_test_picture_window);

	static bool show_window = true;
	static set_detect_info detect_info;

	//图片检测控制
	if (ImGui::CollapsingHeader(u8"设置"))
	{
		ImGui::InputFloat(u8"thresh", &detect_info.thresh, 0.01f, 1.0f, "%.3f");
		ImGui::InputFloat(u8"hier_thresh", &detect_info.hier_thresh, 0.01f, 1.0f, "%.3f");
		ImGui::InputFloat(u8"nms", &detect_info.nms, 0.01f, 1.0f, "%.3f");

		ImGui::Checkbox(u8"显示窗口", &show_window);
	}

	//颜色控制
	if (ImGui::CollapsingHeader(u8"颜色"))
	{
		ImGui::ColorEdit3(u8"方框颜色", g_global_set.color_set.box_rgb);
		ImGui::ColorEdit3(u8"字体颜色", g_global_set.color_set.font_rgb);
		ImGui::InputFloat(u8"线条厚度", &g_global_set.color_set.thickness, 0.01f, 1.0f, "%.3f");
	}

	const ImVec2 size = ImGui::GetContentRegionAvail();

	static char target_picture[default_char_size] = "match.jpg";
	ImGui::InputText(u8"测试图片", (char*)string_to_utf8(target_picture).c_str(), default_char_size);
	if (ImGui::Button(u8"选择图片", ImVec2(size.x / 2, 0.0f))) select_type_file("image file\0*.jpg;0*.bmp;0*.png\0\0", target_picture, default_char_size);
	ImGui::SameLine();

	if (ImGui::Button(u8"执行检测", ImVec2(size.x / 2, 0.0f)))
	{
		if (g_global_set.net_set.initizlie) analyse_picture(target_picture, detect_info, show_window);
		else show_window_tip("网络没有初始化");
	}

	//存在图像纹理，就显示图片
	if (g_global_set.direct3dtexture9)
		ImGui::Image(g_global_set.direct3dtexture9, ImGui::GetContentRegionAvail());

	ImGui::End();
}

void imgui_test_video_window()
{
	if (!g_global_set.imgui_show_set.show_test_video_window) return;

	ImGui::SetNextWindowSize(ImVec2(500, 700), ImGuiCond_FirstUseEver);
	ImGui::Begin(u8"智能交通系统  -  测试视频窗口",&g_global_set.imgui_show_set.show_test_video_window);

	//视频控制
	static video_control control_info;
	if (ImGui::CollapsingHeader(u8"延迟"))
	{
		ImGui::InputInt(u8"显示延迟", &control_info.show_delay);
		ImGui::InputInt(u8"读取延迟", &control_info.read_delay);
		ImGui::InputInt(u8"检测延迟", &control_info.detect_delay);
		ImGui::InputInt(u8"场景延迟", &control_info.scene_delay);
	}

	if (ImGui::CollapsingHeader(u8"检测"))
	{
		ImGui::InputFloat(u8"thresh", &g_global_set.video_detect_set.thresh, 0.1f, 1.0f, "%.2f");
		ImGui::InputFloat(u8"hier_thresh", &g_global_set.video_detect_set.hier_thresh, 0.1f, 1.0f, "%.2f");
		ImGui::InputFloat(u8"nms", &g_global_set.video_detect_set.nms, 0.1f, 1.0f, "%.2f");
		ImGui::InputInt(u8"检测线程", &control_info.detect_count);
	}

	if (ImGui::CollapsingHeader(u8"场景"))
	{
		ImGui::Checkbox(u8"统计人流量",&g_global_set.secne_set.human_traffic);
		ImGui::Checkbox(u8"统计车流量", &g_global_set.secne_set.car_traffic);
		ImGui::Checkbox(u8"占用公交车道", &g_global_set.secne_set.occupy_bus_lane);
		ImGui::Checkbox(u8"闯红灯", &g_global_set.secne_set.rush_red_light);
		ImGui::Checkbox(u8"不按导向行驶", &g_global_set.secne_set.not_guided);
		ImGui::Checkbox(u8"斑马线不礼让行人", &g_global_set.secne_set.not_zebra_cross);
	}

	if (ImGui::CollapsingHeader(u8"颜色"))
	{
		ImGui::ColorEdit3(u8"方框颜色", g_global_set.color_set.box_rgb);
		ImGui::ColorEdit3(u8"字体颜色", g_global_set.color_set.font_rgb);
		ImGui::InputFloat(u8"线条厚度", &g_global_set.color_set.thickness, 0.01f, 1.0f, "%.3f");
	}

	ImGui::RadioButton(u8"视频模式", &control_info.use_camera,0);
	ImGui::SameLine();
	ImGui::RadioButton(u8"摄像头模式", &control_info.use_camera,1);

	if (control_info.use_camera) ImGui::InputInt(u8"摄像头索引", &control_info.camera_index);
	else
	{
		ImGui::InputText(u8"视频路径", (char*)string_to_utf8(control_info.video_path).c_str(), default_char_size);
		if (ImGui::Button(u8"选择视频", ImVec2(-FLT_MIN, 0.0f))) select_type_file("video file\0*.mp4;*.avi;*.flv;*.ts\0\0", control_info.video_path);
	}

	const ImVec2 size = ImGui::GetContentRegionAvail();

	if (ImGui::Button(u8"标记区域", ImVec2(size.x / 3, 0.0f))) g_global_set.imgui_show_set.show_set_load_region_window = true;
	ImGui::SameLine();
	if (ImGui::Button(u8"开始检测", ImVec2(size.x / 3, 0.0f)))
	{
		if (!control_info.leave) show_window_tip("请先停止上一个视频的检测");
		else if (g_global_set.net_set.initizlie) _beginthreadex(NULL, 0, analyse_video, &control_info, 0, NULL);
		else show_window_tip("网络没有初始化");
	}
	ImGui::SameLine();
	if (ImGui::Button(u8"停止检测", ImVec2(size.x / 3, 0.0f))) control_info.leave = true;
	
	ImGui::Columns(2, NULL, TRUE);
	ImGui::Separator();
	ImGui::BulletText(u8"总人流量 : %d ", g_global_set.secne_set.human_count);
	ImGui::BulletText(u8"当前人流量 : %d ", g_global_set.secne_set.human_current);
	ImGui::BulletText(u8"平均人流量 : %d ", g_global_set.secne_set.human_avg);
	if (ImGui::Button(u8"全部重置", ImVec2(-FLT_MIN, 0.0f)))
	{
		g_global_set.secne_set.human_count = 0;
		g_global_set.secne_set.human_current = 0;
		g_global_set.secne_set.human_avg = 0;
	}
	ImGui::NextColumn();

	ImGui::BulletText(u8"总车流量 : %d ", g_global_set.secne_set.car_count);
	ImGui::BulletText(u8"当前车流量 : %d ", g_global_set.secne_set.car_current);
	ImGui::BulletText(u8"平均车流量 : %d ", g_global_set.secne_set.car_avg);
	if (ImGui::Button(u8"全部重置", ImVec2(-FLT_MIN, 0.0f)))
	{
		g_global_set.secne_set.car_count = 0;
		g_global_set.secne_set.car_current = 0;
		g_global_set.secne_set.car_avg = 0;
	}
	ImGui::Columns(1);
	ImGui::Separator();

	if (g_global_set.secne_set.occupy_bus_lane && g_global_set.secne_set.occupy_bus_list.size())
	{
		//队列只显示5个，多的可以保存道文件里面
		int size = g_global_set.secne_set.occupy_bus_list.size();
		if (size > 5)
			g_global_set.secne_set.occupy_bus_list.erase(g_global_set.secne_set.occupy_bus_list.begin(),
				g_global_set.secne_set.occupy_bus_list.begin() + (size - 5));

		//显示
		for (auto& it : g_global_set.secne_set.occupy_bus_list)
			ImGui::BulletText(u8"%d年%d月%d日%d时%d分%d秒 占用公交车道",
				it.times[0], it.times[1], it.times[2], it.times[3], it.times[4], it.times[5]);
		ImGui::Separator();
	}


	ImGui::End();
}

void imgui_load_region_window()
{
	if (!g_global_set.imgui_show_set.show_set_load_region_window) return;

	ImGui::SetNextWindowSize(ImVec2{ 600,600 }, ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowPos(ImVec2{ 0,0 });
	ImGui::Begin(u8"region set", &g_global_set.imgui_show_set.show_set_load_region_window, ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoMove);

	//获取绘制句柄
	ImDrawList* draw_list = ImGui::GetWindowDrawList();

	//方框颜色
	static ImVec4 colf = ImVec4(0.0f, 1.0f, 0.0f, 1.0f);
	const ImU32 col = ImColor(colf);

	//方框信息
	static region_mask region_info;

	//步骤
	static int set_step = 0;

	//开始位置
	static ImVec2 start_pos{ -1,-1 };

	//控制相关
	if (ImGui::BeginMenuBar())
	{
		static char video_path[default_char_size];
		if (ImGui::BeginMenu(u8"读取一帧视频"))
		{
			if (ImGui::Button(u8"read") && select_type_file("video file\0*.mp4;0*.avi\0\0", video_path, default_char_size))
			{
				read_video_frame(video_path);
				set_step = 0;
			}
			ImGui::EndMenu();
		}
		if (ImGui::BeginMenu(u8"标记斑马线"))
		{
			if (ImGui::Button(u8"set"))
			{
				colf.w = colf.y = 1.0f; colf.x = colf.z = 0;
				region_info.type = region_zebra_cross;
				set_step = 0;
			}
			ImGui::EndMenu();
		}
		if (ImGui::BeginMenu(u8"标记公交车专用道"))
		{
			if (ImGui::Button(u8"set"))
			{
				colf.w = colf.z = 1.0f; colf.y = colf.x = 0;
				region_info.type = region_bus_lane;
				set_step = 0;
			}
			ImGui::EndMenu();
		}
		if (ImGui::BeginMenu(u8"标记路边车位"))
		{
			if (ImGui::Button(u8"set"))
			{
				colf.w = colf.x = 1.0f; colf.y = colf.z = 0;
				region_info.type = region_street_parking;
				set_step = 0;
			}
			ImGui::EndMenu();
		}
		if (ImGui::BeginMenu(u8"使用说明"))
		{
			ImGui::BulletText(u8"读取一帧视频");
			ImGui::BulletText(u8"选择方框标记类型");
			ImGui::BulletText(u8"单击鼠标左键设置开始位置");
			ImGui::BulletText(u8"再次单击鼠标左键设置结束位置");
			ImGui::BulletText(u8"单击鼠标右键删除上一个方框标记");
			ImGui::EndMenu();
		}
		ImGui::EndMenuBar();
	}

	//获取位置
	const ImVec2 pos = ImGui::GetCursorScreenPos();
	const ImVec2 size = ImGui::GetContentRegionAvail();

	//先绘制图片
	if (g_global_set.direct3dtexture9)
		ImGui::Image(g_global_set.direct3dtexture9, size);

	//获取当前鼠标在当前窗口的位置
	ImVec2 current_pos = { ImGui::GetIO().MousePos.x - pos.x + 8,ImGui::GetIO().MousePos.y - pos.y + 60};
	//ImGui::Text(u8"区域X轴 %f   区域Y轴 %f", pos.x, pos.y);
	//ImGui::Text(u8"区域宽度 %f  区域高度 %f", size.x, size.y);
	ImGui::Text(u8"窗口宽度 %f  窗口高度 %f", ImGui::GetWindowWidth(), ImGui::GetWindowHeight());
	ImGui::Text(u8"窗口X轴  %f  窗口Y轴  %f", ImGui::GetWindowPos().x, ImGui::GetWindowPos().y);
	ImGui::Text(u8"鼠标X轴  %f  鼠标Y轴  %f", ImGui::GetIO().MousePos.x, ImGui::GetIO().MousePos.y);
	//ImGui::Text(u8"%f %f", current_pos.x, current_pos.y - 50.0f);//这里才是原点啊 (0,0)
	

	//鼠标左键按下
	if (ImGui::IsMouseClicked(0))
	{
		//不在区域位置不画图
		if (current_pos.x <= 0 || current_pos.y - 50.0f <= 0 || current_pos.x - 8 >= size.x || current_pos.y - 60 >= size.y) {}
		else if (set_step == 0)//设置开始位置
		{
			start_pos = current_pos;
			set_step++;
		}
		else if (set_step == 1)//保存进入列表
		{
			//不能太小
			if (abs(current_pos.x - start_pos.x) < 10.0f && abs(current_pos.y - start_pos.y) < 10.0f) {}
			else
			{
				region_info.window_size = size;
				region_info.pos = start_pos;
				region_info.size = current_pos;
				region_info.rect_color = colf;
				g_global_set.mask_list.push_back(region_info);
				start_pos = { -1,-1 };
				set_step = 0;
			}
		}
	}

	//鼠标右键按下
	if (ImGui::IsMouseClicked(1))
	{
		if (!set_step && g_global_set.mask_list.size()) g_global_set.mask_list.pop_back();
		else
		{
			set_step = 0;
			start_pos = { -1,-1 };
		}
	}

	//显示队列中的方框
	for (auto& it : g_global_set.mask_list) draw_list->AddRect(it.pos, it.size, ImColor(it.rect_color), 0.0f, 0, 1.0f);

	//绘制方框
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
}

