#include "gui.h"

gui* gui::m_static_this = nullptr;

r_hresult __stdcall gui::window_proc(r_hwnd _hwnd, r_uint msg, r_wparam wpa, r_lparam lpa) noexcept
{
	extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
	if (ImGui_ImplWin32_WndProcHandler(_hwnd, msg, wpa, lpa)) return true;

	switch (msg)
	{
	case WM_SIZE:
		if (gui::m_static_this) gui::m_static_this->notice_reset(LOWORD(lpa), HIWORD(lpa));
		break;
	case WM_CLOSE:
		PostQuitMessage(0);
		break;
	default: return DefWindowProcA(_hwnd, msg, wpa, lpa);
	}
	return 1;
}

void gui::initialize_d3d9() noexcept
{
	m_IDirect3D9 = Direct3DCreate9(D3D_SDK_VERSION);
	check_error(m_IDirect3D9, "Direct3DCreate9函数失败");

	memset(&m_D3DPRESENT_PARAMETERS, 0, sizeof(m_D3DPRESENT_PARAMETERS));
	m_D3DPRESENT_PARAMETERS.Windowed = TRUE;
	m_D3DPRESENT_PARAMETERS.SwapEffect = D3DSWAPEFFECT_DISCARD;
	m_D3DPRESENT_PARAMETERS.BackBufferFormat = D3DFMT_UNKNOWN;
	m_D3DPRESENT_PARAMETERS.EnableAutoDepthStencil = TRUE;
	m_D3DPRESENT_PARAMETERS.AutoDepthStencilFormat = D3DFMT_D16;
	m_D3DPRESENT_PARAMETERS.PresentationInterval = D3DPRESENT_INTERVAL_ONE;
	r_hresult result = m_IDirect3D9->CreateDevice(D3DADAPTER_DEFAULT, D3DDEVTYPE_HAL, m_hwnd, D3DCREATE_HARDWARE_VERTEXPROCESSING, &m_D3DPRESENT_PARAMETERS, &m_IDirect3DDevice9);
	check_error(result == S_OK, "CreateDevice函数失败");
}

void gui::initialize_imgui() noexcept
{
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui::StyleColorsDark();
	//ImGui::GetStyle().FrameRounding = 120.f;
	//ImGui::GetStyle().GrabRounding = 12.0f;
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	io.IniFilename = nullptr;
	io.LogFilename = nullptr;
	io.Fonts->AddFontFromFileTTF("C:\\msyh.ttc", 20.0f, NULL, io.Fonts->GetGlyphRangesChineseFull());

	ImGui_ImplWin32_Init(m_hwnd);
	ImGui_ImplDX9_Init(m_IDirect3DDevice9);
}

void gui::render_handle() noexcept
{
	ImGui_ImplDX9_NewFrame();
	ImGui_ImplWin32_NewFrame();
	ImGui::NewFrame();

	imgui_window_meun();//菜单
	imgui_display_video();//视频显示
	//ImGui::ShowDemoWindow();

	ImGui::EndFrame();
	m_IDirect3DDevice9->SetRenderState(D3DRS_ZENABLE, FALSE);
	m_IDirect3DDevice9->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);
	m_IDirect3DDevice9->SetRenderState(D3DRS_SCISSORTESTENABLE, FALSE);
	m_IDirect3DDevice9->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, D3DCOLOR_RGBA(255, 255, 255, 255), 1.0f, 0);

	if (m_IDirect3DDevice9->BeginScene() >= 0)
	{
		ImGui::Render();
		ImGui_ImplDX9_RenderDrawData(ImGui::GetDrawData());
		m_IDirect3DDevice9->EndScene();
	}

	bool state = m_IDirect3DDevice9->Present(NULL, NULL, NULL, NULL) == D3DERR_DEVICELOST;
	if (state)
	{
		state = m_IDirect3DDevice9->TestCooperativeLevel() == D3DERR_DEVICENOTRESET;
		if (state) device_reset();
	}

	if (m_IDirect3DTexture9)
	{
		m_IDirect3DTexture9->Release();
		m_IDirect3DTexture9 = nullptr;
	}
}

void gui::device_reset() noexcept
{
	if (m_IDirect3DDevice9)
	{
		ImGui_ImplDX9_InvalidateDeviceObjects();
		r_hresult  result = m_IDirect3DDevice9->Reset(&m_D3DPRESENT_PARAMETERS);
		check_error(result != D3DERR_INVALIDCALL, "设备无效");
		ImGui_ImplDX9_CreateDeviceObjects();
	}
}

void gui::clear_d3d9() noexcept
{
	if (m_IDirect3D9) m_IDirect3D9->Release();
	if (m_IDirect3DDevice9) m_IDirect3DDevice9->Release();
	m_IDirect3D9 = nullptr;
	m_IDirect3DDevice9 = nullptr;
}

void gui::clear_imgui() noexcept
{
	ImGui_ImplDX9_Shutdown();
	ImGui_ImplWin32_Shutdown();
	ImGui::DestroyContext();
}

char* gui::to_utf8(const char* text, char* buffer, int size) noexcept
{
	int text_len = strlen(text);
	if (text_len > size) return buffer;

	wchar_t* temp = new wchar_t[text_len];
	if (temp == 0) return buffer;

	memset(temp, 0, sizeof(wchar_t) * text_len);
	memset(buffer, 0, sizeof(char)* size);
	MultiByteToWideChar(CP_ACP, 0, text, text_len, temp, sizeof(wchar_t) * text_len);
	WideCharToMultiByte(CP_UTF8, 0, temp, text_len, buffer, size, nullptr, nullptr);

	delete[] temp;
	return buffer;
}

void gui::update_texture(struct frame_handle* data) noexcept
{
	int w = data->frame.cols;
	int h = data->frame.rows;
	int c = data->frame.channels();

	unsigned char* buffer = new unsigned char[w * h * (c + 1)];
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			buffer[(i * w + j) * (c + 1) + 0] = data->frame.data[(i * w + j) * c + 0];
			buffer[(i * w + j) * (c + 1) + 1] = data->frame.data[(i * w + j) * c + 1];
			buffer[(i * w + j) * (c + 1) + 2] = data->frame.data[(i * w + j) * c + 2];
			buffer[(i * w + j) * (c + 1) + 3] = 0xff;
		}
	}

	r_hresult result = m_IDirect3DDevice9->CreateTexture(w, h, 1, D3DUSAGE_DYNAMIC, D3DFMT_X8R8G8B8, D3DPOOL_DEFAULT, &m_IDirect3DTexture9, nullptr);
	if (result == S_OK)
	{
		D3DLOCKED_RECT rect{ 0 };
		m_IDirect3DTexture9->LockRect(0, &rect, nullptr, 0);

		for (int i = 0; i < h; i++)
			memcpy((unsigned char*)rect.pBits + rect.Pitch * i, buffer + (w * (c + 1)) * i, w * (c + 1));

		m_IDirect3DTexture9->UnlockRect(0);
	}

	data->frame.release();
	delete data;
	delete buffer;
}

void gui::imgui_display_video() noexcept
{
	char z_title[max_string_len] = "video display";
	if (m_is_english == false)
	{
		to_utf8("视频播放", z_title, max_string_len);
	}

	ImGui::SetNextWindowSize(ImVec2{ 800.0f,300.0f }, ImGuiCond_FirstUseEver);
	ImGui::Begin(z_title);

	struct frame_handle* data = m_video.get_video_frame();
	if (data)
	{
		update_texture(data);
		if (m_IDirect3DTexture9) ImGui::Image(m_IDirect3DTexture9, ImGui::GetContentRegionAvail());
	}

	ImVec2 pos = ImGui::GetWindowPos();
	pos.y += ImGui::GetWindowHeight();

	this->imgui_video_control_overlay(pos, ImGui::GetWindowWidth());
	ImGui::End();
}

void gui::imgui_video_control_overlay(ImVec2 pos, float width) noexcept
{
	bool state = true;
	ImGui::SetNextWindowBgAlpha(0.35f);
	ImGui::SetNextWindowPos(pos);
	ImGui::SetNextWindowSize(ImVec2{ width,80.0f });
	ImGui::Begin("controls", &state, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav);

	char z_pause[max_string_len] = "pause";
	char z_display[max_string_len] = "display";
	char z_exit[max_string_len] = "exit";
	char z_schedule[max_string_len] = "schedule";
	char z_fps[max_string_len] = "FPS : %.2lf";
	if (m_is_english == false)
	{
		to_utf8("暂停", z_pause, max_string_len);
		to_utf8("播放", z_display, max_string_len);
		to_utf8("退出", z_exit, max_string_len);
		to_utf8("进度", z_schedule, max_string_len);
	}

	ImVec2 button_size{ 80.0f,30.f };

	if (ImGui::Button(z_pause, button_size)) m_video.pause();
	ImGui::SameLine();
	if (ImGui::Button(z_display, button_size)) m_video.restart();
	ImGui::SameLine();
	if (ImGui::Button(z_exit, button_size)) m_video.close();
	ImGui::SameLine();

	float value = m_video.get_finish_rate() * 100.0f;
	if (ImGui::SliderFloat(z_schedule, &value, 0.0f, 100.0f))
		m_video.set_frame_index(value / 100.0f);

	ImGui::Text(z_fps, m_video.get_display_fps());

	ImGui::End();
}

void gui::imgui_window_meun() noexcept
{
	if (ImGui::BeginMainMenuBar())
	{
		imgui_file_window();
		imgui_model_window();
		imgui_win_window();
		imgui_mode_window();
		imgui_language_window();
		ImGui::EndMainMenuBar();
	}
}

void gui::imgui_file_window() noexcept
{
	char z_file[max_string_len] = "File";
	char z_load_file[max_string_len] = "Load File";
	if (m_is_english == false)
	{
		to_utf8("文件", z_file, max_string_len);
		to_utf8("选择文件", z_load_file, max_string_len);
	}

	if (ImGui::BeginMenu(z_file))
	{
		char buffer[max_string_len]{ 0 };
		if (ImGui::MenuItem(z_load_file))
		{
			get_file_path("Video or Image \0*.mp4;*.flv;*.ts;*.jpg;*.bmp;*.png\0\0", buffer, max_string_len);
			switch (get_file_type(buffer))
			{
			case 1:
				m_video.set_video_path(buffer);
				m_video.start();
				break;
			case 2:
				break;
			}
		}
		ImGui::EndMenu();
	}
}

void gui::imgui_model_window() noexcept
{
	char z_model[max_string_len] = "Model";
	char z_detect_model[max_string_len] = "Object detection model";
	char z_recognition_model[max_string_len] = "License plate recognition model";
	char z_Load_model[max_string_len] = "Load model";
	char z_Unload_model[max_string_len] = "Unload model";
	if (m_is_english == false)
	{
		to_utf8("模型", z_model, max_string_len);
		to_utf8("物体检测模型", z_detect_model, max_string_len);
		to_utf8("车牌识别系统", z_recognition_model, max_string_len);
		to_utf8("加载模型", z_Load_model, max_string_len);
		to_utf8("卸载模型", z_Unload_model, max_string_len);
	}

	if (ImGui::BeginMenu(z_model))
	{
		if (ImGui::BeginMenu(z_detect_model))
		{
			if (ImGui::MenuItem(z_Load_model)) {}
			if (ImGui::MenuItem(z_Unload_model)) {}
			ImGui::EndMenu();
		}
		if (ImGui::BeginMenu(z_recognition_model))
		{
			if (ImGui::MenuItem(z_Load_model)) {}
			if (ImGui::MenuItem(z_Unload_model)) {}
			ImGui::EndMenu();
		}
		ImGui::EndMenu();
	}
}

void gui::imgui_win_window() noexcept
{
	if (ImGui::BeginMenu("Windows"))
	{
		ImGui::EndMenu();
	}
}

void gui::imgui_mode_window() noexcept
{
	if (ImGui::BeginMenu("Mode"))
	{
		ImGui::EndMenu();
	}
}

void gui::imgui_language_window() noexcept
{
	if (ImGui::BeginMenu("Language"))
	{
		char text[1024]{ 0 };
		if (ImGui::MenuItem("English display"))  set_english_display(true);
		if (ImGui::MenuItem(to_utf8("中文显示", text, 1024))) set_english_display(false);
		ImGui::EndMenu();
	}
}

gui::gui() noexcept
{
	m_IDirect3D9 = nullptr;
	m_IDirect3DDevice9 = nullptr;
	m_IDirect3DTexture9 = nullptr;
}

void gui::create_and_show(const char* name /*= "darknet_imgui"*/) noexcept
{
	WNDCLASSEXA window_class{ 0 };
	window_class.cbSize = sizeof(window_class);
	window_class.hbrBackground = HBRUSH(GetStockObject(WHITE_BRUSH));
	window_class.hCursor = LoadCursorA(nullptr, IDC_ARROW);
	window_class.hInstance = GetModuleHandleA(nullptr);
	//window_class.hIcon = LoadIconA(nullptr, "match.ico");
	window_class.lpfnWndProc = (WNDPROC)window_proc;
	window_class.lpszClassName = name;
	window_class.style = CS_VREDRAW | CS_HREDRAW;
	check_error(RegisterClassExA(&window_class), "注册类失败");

	m_hwnd = CreateWindowA(name, name, WS_OVERLAPPEDWINDOW,
		100, 100, 1200, 600, 0, 0, GetModuleHandleA(nullptr), 0);
	check_error(m_hwnd, "窗口创建失败");

	initialize_d3d9();
	initialize_imgui();

	ShowWindow(m_hwnd, SW_SHOW);
	UpdateWindow(m_hwnd);
	gui::m_static_this = this;
}

void gui::msg_handle() noexcept
{
	MSG msg{ 0 };
	while (msg.message != WM_QUIT)
	{
		if (::PeekMessage(&msg, NULL, 0U, 0U, PM_REMOVE))
		{
			::TranslateMessage(&msg);
			::DispatchMessage(&msg);
			continue;
		}
		render_handle();
	}
	clear_imgui();
	clear_d3d9();
}

void gui::notice_reset(int w, int h) noexcept
{
	m_D3DPRESENT_PARAMETERS.BackBufferWidth = w;
	m_D3DPRESENT_PARAMETERS.BackBufferHeight = h;
	device_reset();
}

void gui::set_english_display(bool enable) noexcept
{
	m_is_english = enable;
}
