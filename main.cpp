#include "main.h"

global_set g_global_set;

//int main(int argc, char* argv[])
//{
//	//设置显卡工作
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
//	printf("标记完成!------------------------------------------------------");
//	getchar();
//	return 0;
//}

int main(int argc, char* argv[])//cmd窗口测试作用
////int _stdcall WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpCmdLine, _In_ int nShowCmd)
{
	//设置工作显卡和工作模式
	cuda_set_device(cuda_get_device());
	CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));

	register_window_struct();//注册窗口类
	create_window();//创建窗口
	initialize_d3d9();//初始化d3d设备
	initialize_imgui();//初始化imgui界面库
	window_message_handle();//窗口消息处理
	clear_imgui_set();//清理imgui界面库
	clear_d3d9_set();//清理d3d设备
	return 0;
}

//声明默认的imgui界面库窗口过程
extern LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
LRESULT _stdcall window_process(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam)
{
	//imgui界面库处理了的消息我们不做处理
	if (ImGui_ImplWin32_WndProcHandler(hwnd, msg, wparam, lparam)) return true;

	//窗口大小改变会导致设备丢失，所以我们获取窗口大新大小后要重置d3d设备对象
	if (msg == WM_SIZE && wparam != SIZE_MINIMIZED)
	{
		g_global_set.d3dpresent.BackBufferWidth = LOWORD(lparam);
		g_global_set.d3dpresent.BackBufferHeight = HIWORD(lparam);
		reset_d3d9_device();
		return true;
	}

	//退出程序消息
	if (msg == WM_DESTROY) PostQuitMessage(0);

	//上面没人处理就默认处理
	return DefWindowProcA(hwnd, msg, wparam, lparam);
}

void register_window_struct()
{
	//初始化一个随机种子
	srand((unsigned int)time(0));

	//设置窗口的标题
	sprintf(g_global_set.window_class_name, "基于计算机视觉的交通场景智能应用__%d", rand());

	//初始化窗口类
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

	//注册窗口类
	check_serious_error(RegisterClassExA(&window_class), "注册窗口类失败");
}

void create_window()
{
	//尝试创建窗口
	g_global_set.window_hwnd = CreateWindowExA(NULL, g_global_set.window_class_name, g_global_set.window_class_name,
		WS_OVERLAPPEDWINDOW, 100, 100, 1000, 600, NULL, NULL, GetModuleHandleA(NULL), NULL);
	check_serious_error(g_global_set.window_hwnd, "创建窗口失败");

	//将创建好的窗口进行显示
	ShowWindow(g_global_set.window_hwnd, SW_SHOWMAXIMIZED);
	UpdateWindow(g_global_set.window_hwnd);
}

void initialize_d3d9()
{
	//获取d3d指针
	g_global_set.direct3d9 = Direct3DCreate9(D3D_SDK_VERSION);
	check_serious_error(g_global_set.direct3d9, "初始化d3d9失败");

	//获取d3d9设备指针
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
	IMGUI_CHECKVERSION();//检测imgui界面库版本
	ImGui::CreateContext();//创建一个界面上下文
	ImGuiIO& io = ImGui::GetIO(); (void)io;

	//不保存任何配置文件
	io.IniFilename = NULL;
	io.LogFilename = NULL;

	//设置界面风格为白色
	ImGui::StyleColorsLight();

	//绑定操作
	ImGui_ImplWin32_Init(g_global_set.window_hwnd);
	ImGui_ImplDX9_Init(g_global_set.direct3ddevice9);

	//加载微软雅黑字体，为了支持中文的显示
	const char* font_path = "msyh.ttc";
	check_serious_error(_access(font_path, 0) != -1, "微软字体字体文件缺失");
	io.Fonts->AddFontFromFileTTF(font_path, 20.0f, NULL, io.Fonts->GetGlyphRangesChineseFull());
}

void window_message_handle()
{
	//不断获取窗口消息，没有窗口消息就进行界面的绘制操作
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
	//存在纹理就释放
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
	imgui_object_detect_window();
	imgui_car_id_identify_window();
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
	//设置窗口的大小和位置
	ImGui::SetNextWindowSize(ImVec2(500, 300), ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowPos(ImVec2(0, 0));
	ImGui::Begin(u8"智能交通系统");

	//获取显卡数量
	static int gpu_count = cuda_get_device();

	//获取显卡相关信息
	static cudaDeviceProp* gpu_info = get_gpu_infomation(gpu_count);

	//显示显卡相关信息
	for (int i = 0; i < gpu_count; i++)
		ImGui::BulletText(u8"GPU型号\t %s", gpu_info[i].name);

	//获取系统类型
	static int os_type = get_os_type();
	if (os_type == -1) ImGui::BulletText(u8"系统类型\t 未知");
	else ImGui::BulletText(u8"系统类型\t Windows %d", os_type);

	//显示CPU核心数
	static int cpu_kernel = get_cpu_kernel();
	ImGui::BulletText(u8"CPU核心\t %d", cpu_kernel);

	//显示内存总数
	static int phy_memory = get_physical_memory();
	ImGui::BulletText(u8"物理内存\t %d", phy_memory);

	ImGui::Separator();
	if (ImGui::Button(u8"物体检测模型配置")) g_global_set.imgui_show_set.show_object_detect_window = true;
	ImGui::SameLine();
	if (ImGui::Button(u8"车牌识别模型配置")) g_global_set.imgui_show_set.show_car_id_identify_window = true;
	if (ImGui::Button(u8"交通图片物体检测")) g_global_set.imgui_show_set.show_test_picture_window = true;
	ImGui::SameLine();
	if (ImGui::Button(u8"交通视频物体检测")) g_global_set.imgui_show_set.show_test_video_window = true;

	ImGui::End();
}

void imgui_object_detect_window()
{
	if (!g_global_set.imgui_show_set.show_object_detect_window) return;
	ImGui::SetNextWindowSize(ImVec2(500, 300), ImGuiCond_FirstUseEver);
	ImGui::Begin(u8"物体检测", &g_global_set.imgui_show_set.show_object_detect_window);

	//获取可用区域大小
	const ImVec2 size = ImGui::GetContentRegionAvail();

	static char names_path[default_char_size] = "object_detect.names";
	ImGui::InputText(u8"*names文件路径", (char*)string_to_utf8((const char*)names_path).c_str(), default_char_size);

	static char cfg_path[default_char_size] = "object_detect.cfg";
	ImGui::InputText(u8"*.cfg文件路径", (char*)string_to_utf8((const char*)cfg_path).c_str(), default_char_size);

	static char weights_path[default_char_size] = "object_detect.weights";
	ImGui::InputText(u8"*.weights文件路径", (char*)string_to_utf8((const char*)weights_path).c_str(), default_char_size);

	if (ImGui::Button(u8"选择*.names文件", ImVec2(size.x / 3, 0.0f))) select_type_file("names flie\0*.names\0\0", names_path);
	ImGui::SameLine();
	if (ImGui::Button(u8"选择*.cfg文件", ImVec2(size.x / 3, 0.0f))) select_type_file("cfg file\0*.cfg\0\0", cfg_path);
	ImGui::SameLine();
	if (ImGui::Button(u8"选择*.weights文件", ImVec2(size.x / 3, 0.0f))) select_type_file("weights file\0*.weights\0\0", weights_path);
	ImGui::Separator();

	ImGui::BulletText(u8"初始化状态 : "); ImGui::SameLine();
	if(g_global_set.object_detect_net_set.initizlie) ImGui::TextColored(ImVec4(255, 0, 0, 255), u8"[ 是 ]");
	else ImGui::TextColored(ImVec4(0, 0, 255, 255), u8"[ 否 ]");

	if (ImGui::Button(u8"初始化物体检测网络", ImVec2(size.x / 2, 0.0f))) initialize_object_detect_net(names_path, cfg_path, weights_path);
	ImGui::SameLine();
	if (ImGui::Button(u8"释放物体检测网络", ImVec2(size.x / 2, 0.0f))) clear_object_detect_net();

	ImGui::End();
}

void imgui_car_id_identify_window()
{
	if (!g_global_set.imgui_show_set.show_car_id_identify_window) return;
	ImGui::SetNextWindowSize(ImVec2(500, 300), ImGuiCond_FirstUseEver);
	ImGui::Begin(u8"车牌识别", &g_global_set.imgui_show_set.show_car_id_identify_window);

	//获取可用区域大小
	const ImVec2 size = ImGui::GetContentRegionAvail();

	static char names_path[default_char_size] = "car_id.names";
	ImGui::InputText(u8"*names文件路径", (char*)string_to_utf8((const char*)names_path).c_str(), default_char_size);

	static char cfg_path[default_char_size] = "car_id.cfg";
	ImGui::InputText(u8"*.cfg文件路径", (char*)string_to_utf8((const char*)cfg_path).c_str(), default_char_size);

	static char weights_path[default_char_size] = "car_id.weights";
	ImGui::InputText(u8"*.weights文件路径", (char*)string_to_utf8((const char*)weights_path).c_str(), default_char_size);

	if (ImGui::Button(u8"选择*.names文件", ImVec2(size.x / 3, 0.0f))) select_type_file("names flie\0*.names\0\0", names_path);
	ImGui::SameLine();
	if (ImGui::Button(u8"选择*.cfg文件", ImVec2(size.x / 3, 0.0f))) select_type_file("cfg file\0*.cfg\0\0", cfg_path);
	ImGui::SameLine();
	if (ImGui::Button(u8"选择*.weights文件", ImVec2(size.x / 3, 0.0f))) select_type_file("weights file\0*.weights\0\0", weights_path);
	ImGui::Separator();

	ImGui::BulletText(u8"初始化状态 : ");
	ImGui::SameLine();
	if (g_global_set.car_id_identify_net.initizlie) ImGui::TextColored(ImVec4(255, 0, 0, 255), u8"[ 是 ]");
	else ImGui::TextColored(ImVec4(0, 0, 255, 255), u8"[ 否 ]");

	if (ImGui::Button(u8"初始化车牌识别网络", ImVec2(size.x / 2, 0.0f))) initialize_car_id_identify_net(names_path, cfg_path, weights_path);
	ImGui::SameLine();
	if (ImGui::Button(u8"释放车牌识别网络", ImVec2(size.x / 2, 0.0f))) clear_car_id_identify_net();

	ImGui::End();
}

void imgui_test_picture_window()
{
	if (!g_global_set.imgui_show_set.show_test_picture_window) return;

	ImGui::SetNextWindowSize(ImVec2(300, 300), ImGuiCond_FirstUseEver);
	ImGui::Begin(u8"智能交通系统  -  测试图片窗口", &g_global_set.imgui_show_set.show_test_picture_window);

	static set_detect_info detect_info;

	//图片检测控制
	if (ImGui::CollapsingHeader(u8"设置"))
	{
		ImGui::InputFloat(u8"thresh", &detect_info.thresh, 0.01f, 1.0f, "%.3f");
		ImGui::InputFloat(u8"hier_thresh", &detect_info.hier_thresh, 0.01f, 1.0f, "%.3f");
		ImGui::InputFloat(u8"nms", &detect_info.nms, 0.01f, 1.0f, "%.3f");
	}

	//颜色控制
	if (ImGui::CollapsingHeader(u8"颜色"))
	{
		ImGui::ColorEdit3(u8"方框颜色", g_global_set.color_set.box_rgb);
		ImGui::ColorEdit3(u8"字体颜色", g_global_set.color_set.font_rgb);
		ImGui::InputFloat(u8"线条厚度", &g_global_set.color_set.thickness, 0.01f, 1.0f, "%.3f");
	}

	//获取可用区域大小
	const ImVec2 size = ImGui::GetContentRegionAvail();

	static char target_picture[default_char_size] = "match.jpg";
	ImGui::InputText(u8"测试图片", (char*)string_to_utf8(target_picture).c_str(), default_char_size);
	if (ImGui::Button(u8"选择图片", ImVec2(size.x / 2, 0.0f))) select_type_file("image file\0*.jpg;*.bmp;*.png\0\0", target_picture, default_char_size);
	ImGui::SameLine();

	if (ImGui::Button(u8"执行检测", ImVec2(size.x / 2, 0.0f)))
	{
		if (g_global_set.object_detect_net_set.initizlie) analyse_picture(target_picture, detect_info);
		else show_window_tip("网络没有初始化");
	}

	ImGui::BulletText(u8"物体检测耗时 : %.2lfms ", detect_info.detect_time);
	ImGui::BulletText(u8"车牌识别耗时 : %.2lfms", detect_info.identify_time);

	ImGui::End();
}

void imgui_test_video_window()
{
	if (!g_global_set.imgui_show_set.show_test_video_window) return;

	ImGui::SetNextWindowSize(ImVec2(500, 700), ImGuiCond_FirstUseEver);
	ImGui::Begin(u8"智能交通系统  -  测试视频窗口", &g_global_set.imgui_show_set.show_test_video_window);

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
		ImGui::Checkbox(u8"统计人流量", &g_global_set.secne_set.human_traffic);
		ImGui::Checkbox(u8"统计车流量", &g_global_set.secne_set.car_traffic);
		ImGui::Checkbox(u8"占用公交车道", &g_global_set.secne_set.occupy_bus_lane);
		ImGui::Checkbox(u8"闯红灯", &g_global_set.secne_set.rush_red_light);
		ImGui::Checkbox(u8"不按导向行驶", &g_global_set.secne_set.not_guided);
		ImGui::Checkbox(u8"斑马线不礼让行人", &g_global_set.secne_set.not_zebra_cross);
	}

	if (ImGui::CollapsingHeader(u8"设置"))
	{
		ImGui::ColorEdit3(u8"方框颜色", g_global_set.color_set.box_rgb);
		ImGui::ColorEdit3(u8"字体颜色", g_global_set.color_set.font_rgb);
		ImGui::InputFloat(u8"线条厚度", &g_global_set.color_set.thickness, 0.01f, 1.0f, "%.3f");
		ImGui::InputInt2(u8"视频大小", control_info.video_size);
	}

	ImGui::RadioButton(u8"视频模式", &control_info.use_camera, 0);
	ImGui::SameLine();
	ImGui::RadioButton(u8"摄像头模式", &control_info.use_camera, 1);

	if (control_info.use_camera) ImGui::InputInt(u8"摄像头索引", &control_info.camera_index);
	else
	{
		ImGui::InputText(u8"视频路径", (char*)string_to_utf8(control_info.video_path).c_str(), default_char_size);
		if (ImGui::Button(u8"选择视频", ImVec2(-FLT_MIN, 0.0f))) select_type_file("video file\0*.mp4;*.avi;*.flv;*.ts\0\0", control_info.video_path);
	}

	//获取可显示区域大小
	const ImVec2 region_size = ImGui::GetContentRegionAvail();

	if (ImGui::Button(u8"标记区域", ImVec2(region_size.x / 3, 0.0f))) g_global_set.imgui_show_set.show_set_load_region_window = true;
	ImGui::SameLine();
	if (ImGui::Button(u8"开始检测", ImVec2(region_size.x / 3, 0.0f)))
	{
		if (!control_info.leave) show_window_tip("请先停止上一个视频的检测");
		else if (g_global_set.object_detect_net_set.initizlie) _beginthreadex(NULL, 0, analyse_video, &control_info, 0, NULL);
		else show_window_tip("网络没有初始化");
	}
	ImGui::SameLine();
	if (ImGui::Button(u8"停止检测", ImVec2(region_size.x / 3, 0.0f))) control_info.leave = true;
	
	if (g_global_set.secne_set.human_traffic)
	{
		ImGui::Separator();

		//防止0
		int tick_size = g_global_set.secne_set.human_num.size();
		if (!tick_size) tick_size = 1;

		//申请内存
		float *temp = new float[tick_size];
		memset(temp, 0, tick_size * sizeof(float));

		int total_val = 0;
		static int max_val = 0;
		for (int i = 0; i < g_global_set.secne_set.human_num.size(); i++)
		{
			temp[i] = g_global_set.secne_set.human_num[i];
			total_val += temp[i];//计算总数
			if (max_val < temp[i]) max_val = temp[i];//保存最大的
		}

		//计算平均人流量
		g_global_set.secne_set.human_avg = total_val / tick_size;

		ImGui::BulletText(u8"总人流量 : %d ", g_global_set.secne_set.human_count); ImGui::SameLine();
		ImGui::BulletText(u8"当前人流量 : %d ", g_global_set.secne_set.human_current); ImGui::SameLine();
		ImGui::BulletText(u8"平均人流量 : %d ", g_global_set.secne_set.human_avg);

		//绘制曲线图
		ImGui::PlotHistogram(u8"每分钟人数", temp, tick_size, 0, NULL, 0.0f, (float)max_val, ImVec2(region_size.x, 50.0f));
		
		//释放内存
		delete[] temp;

		//重置
		if (ImGui::Button(u8"重置人流量", ImVec2(-FLT_MIN, 0.0f)))
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
		
		//获取数量
		int tick_size = g_global_set.secne_set.car_num.size();
		if (!tick_size) tick_size = 1;
		
		//申请内存
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

		//计算平均值
		g_global_set.secne_set.car_avg = tatol_val / tick_size;

		ImGui::BulletText(u8"总车流量 : %d ", g_global_set.secne_set.car_count); ImGui::SameLine();
		ImGui::BulletText(u8"当前车流量 : %d ", g_global_set.secne_set.car_current); ImGui::SameLine();
		ImGui::BulletText(u8"平均车流量 : %d ", g_global_set.secne_set.car_avg);

		//绘制曲线图
		ImGui::PlotHistogram(u8"每分钟人数", temp, tick_size, 0, NULL, 0.0f, (float)max_val, ImVec2(region_size.x, 50.0f));

		//释放内存
		delete temp;

		if (ImGui::Button(u8"重置车流量", ImVec2(-FLT_MIN, 0.0f)))
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
		//队列只显示5个，多的可以保存道文件里面
		int size = g_global_set.secne_set.occupy_bus_list.size();
		if (size > 5)
			g_global_set.secne_set.occupy_bus_list.erase(g_global_set.secne_set.occupy_bus_list.begin(),
				g_global_set.secne_set.occupy_bus_list.begin() + (size - 5));

		//显示
		for (auto& it : g_global_set.secne_set.occupy_bus_list)
			ImGui::BulletText(u8"占用公交车道 - %d年%d月%d日%d时%d分%d秒",
				it.times[0], it.times[1], it.times[2], it.times[3], it.times[4], it.times[5]);
		ImGui::Separator();
	}

	ImGui::End();
}

void imgui_load_region_window()
{
	if (!g_global_set.imgui_show_set.show_set_load_region_window) return;

	//设置窗口得大小和位置
	ImGui::SetNextWindowSize(ImVec2{ 600,600 }, ImGuiCond_FirstUseEver);
	ImGui::SetNextWindowPos(ImVec2{ 0,0 });
	ImGui::Begin(u8"区域标记", &g_global_set.imgui_show_set.show_set_load_region_window, ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoMove);

	//获取绘制句柄
	ImDrawList* draw_list = ImGui::GetWindowDrawList();

	//获取大小和位置
	const ImVec2 window_size = { ImGui::GetWindowWidth(),ImGui::GetWindowHeight() };
	const ImVec2 current_pos = { ImGui::GetIO().MousePos.x,ImGui::GetIO().MousePos.y };

	//方框颜色
	static ImVec4 colf = ImVec4(0.0f, 1.0f, 0.0f, 1.0f);
	const ImU32 col = ImColor(colf);

	//方框信息
	static region_mask region_info;

	//步骤 和 开始位置
	static int set_step = 0;
	static ImVec2 start_pos{ -1,-1 };

	//清空 重新开始
	auto clear_start_pos = []()
	{
		set_step = 0;
		start_pos = { -1,-1 };
	};

	//先绘制图片
	if (g_global_set.direct3dtexture9)
		ImGui::Image(g_global_set.direct3dtexture9, ImGui::GetContentRegionAvail());

	//控制相关
	if (ImGui::BeginMenuBar())
	{
		static char video_path[default_char_size];
		if (ImGui::BeginMenu(u8"操作相关"))
		{
			if (ImGui::Button(u8"读取一帧视频图像") && select_type_file("video file\0*.mp4;*.avi;*.flv\0\0", video_path, default_char_size))
			{
				read_video_frame(video_path);
				clear_start_pos();
			}
			if (ImGui::Button(u8"标记斑马线的通道"))
			{
				colf.w = colf.y = 1.0f; colf.x = colf.z = 0;
				region_info.type = region_zebra_cross;
				clear_start_pos();
			}
			if (ImGui::Button(u8"标记公交车专用道"))
			{
				colf.w = colf.z = 1.0f; colf.y = colf.x = 0;
				region_info.type = region_bus_lane;
				clear_start_pos();
			}
			if (ImGui::Button(u8"标记马路边停车位"))
			{
				colf.w = colf.x = 1.0f; colf.y = colf.z = 0;
				region_info.type = region_street_parking;
				clear_start_pos();
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

	//鼠标左键按下
	if (ImGui::IsMouseClicked(0))
	{
		//不在区域位置不画图
		if (current_pos.x <= 0 || current_pos.y - 50.0f <= 0 || current_pos.x >= window_size.x || current_pos.y >= window_size.y) {}
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
				region_info.window_size = { window_size.x,window_size.y - 50.0f };//保存窗口大小
				region_info.pos = start_pos;//保存开始位置
				region_info.size = current_pos;//保存结束
				region_info.rect_color = colf;//保存颜色
				g_global_set.mask_list.push_back(region_info);
				clear_start_pos();
			}
		}
	}

	//鼠标右键按下
	if (ImGui::IsMouseClicked(1))
	{
		if (!set_step && g_global_set.mask_list.size()) g_global_set.mask_list.pop_back();
		else clear_start_pos();
	}

	//显示队列中的方框
	for (auto& it : g_global_set.mask_list) draw_list->AddRect(it.pos, it.size, ImColor(it.rect_color), 0.0f, 0, 1.0f);

	//绘制当前的方框
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

