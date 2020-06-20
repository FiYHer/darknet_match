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
	//ImGui::StyleColorsDark();
	ImGui::StyleColorsLight();
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
	imgui_region_manager();//区域管理
	imgui_people_statistics();//人流量统计
	imgui_car_statistics();//车流量统计
	//ImGui::ShowDemoWindow();

	ImGui::EndFrame();
	m_IDirect3DDevice9->SetRenderState(D3DRS_ZENABLE, FALSE);
	m_IDirect3DDevice9->SetRenderState(D3DRS_ALPHABLENDENABLE, FALSE);
	m_IDirect3DDevice9->SetRenderState(D3DRS_SCISSORTESTENABLE, FALSE);
	m_IDirect3DDevice9->Clear(0, NULL, D3DCLEAR_TARGET | D3DCLEAR_ZBUFFER, D3DCOLOR_RGBA(255, 174, 200, 255), 1.0f, 0);

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

	release_image_texture();
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
	if (data->frame.empty()) return;

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

	delete buffer;
}

IDirect3DTexture9* gui::get_image_texture(cv::Mat* frame) noexcept
{
	if (frame->empty()) return nullptr;

	int w = frame->cols;
	int h = frame->rows;
	int c = frame->channels();

	unsigned char* buffer = new unsigned char[w * h * (c + 1)];
	for (int i = 0; i < h; i++)
	{
		for (int j = 0; j < w; j++)
		{
			buffer[(i * w + j) * (c + 1) + 0] = frame->data[(i * w + j) * c + 0];
			buffer[(i * w + j) * (c + 1) + 1] = frame->data[(i * w + j) * c + 1];
			buffer[(i * w + j) * (c + 1) + 2] = frame->data[(i * w + j) * c + 2];
			buffer[(i * w + j) * (c + 1) + 3] = 0xff;
		}
	}

	IDirect3DTexture9* texture = nullptr;
	r_hresult result = m_IDirect3DDevice9->CreateTexture(w, h, 1, D3DUSAGE_DYNAMIC, D3DFMT_X8R8G8B8, D3DPOOL_DEFAULT, &texture, nullptr);
	if (result == S_OK)
	{
		D3DLOCKED_RECT rect{ 0 };
		texture->LockRect(0, &rect, nullptr, 0);

		for (int i = 0; i < h; i++)
			memcpy((unsigned char*)rect.pBits + rect.Pitch * i, buffer + (w * (c + 1)) * i, w * (c + 1));

		texture->UnlockRect(0);

		//保存进列表 好统一release()
		m_textures_list.push_back(texture);
	}

	delete buffer;

	return texture;
}

void gui::release_image_texture() noexcept
{
	for (auto& it : m_textures_list)
		it->Release();
	m_textures_list.clear();
}

void gui::imgui_display_video() noexcept
{
	char z_title[max_string_len] = "video display";
	if (m_is_english == false)
	{
		to_utf8("视频播放", z_title, max_string_len);
	}

	ImGui::SetNextWindowSize(ImVec2{ 900.0f,500.0f }, ImGuiCond_FirstUseEver);
	ImGui::Begin(z_title);

	static struct frame_handle last_data = {};
	struct frame_handle* data = m_video.get_video_frame();
	if (data)
	{
		if (last_data.frame.empty() == false) last_data.frame.release();
		data->frame.copyTo(last_data.frame);
		data->frame.release();
		delete data;
	}

	update_texture(&last_data);
	if (m_IDirect3DTexture9) ImGui::Image(m_IDirect3DTexture9, ImGui::GetContentRegionAvail());

	ImVec2 pos = ImGui::GetWindowPos();
	pos.y += ImGui::GetWindowHeight();

	this->imgui_select_video(true);
	this->imgui_video_control_overlay(pos, ImGui::GetWindowWidth());
	ImGui::End();
}

void gui::imgui_select_video(bool update) noexcept
{
	static char buffer[max_string_len]{ 0 };

	char z_select[max_string_len] = "select video";
	if (m_is_english == false)
	{
		to_utf8("选择视频", z_select, max_string_len);
	}

	if (ImGui::BeginPopupContextWindow())
	{
		if (ImGui::MenuItem(z_select, nullptr))
		{
			get_file_path("video file\0*.mp4;*.flv;*.ts;*.avi\0\0", buffer, max_string_len);
			int size = strlen(buffer);
			if (size)
			{
				if (update) m_video.set_video_path(buffer);
				m_video.get_per_video_frame(buffer);
			}
		}
		ImGui::EndPopup();
	}
}

void gui::imgui_video_control_overlay(ImVec2 pos, float width) noexcept
{
	static bool state = true;
	ImGui::SetNextWindowBgAlpha(0.35f);
	ImGui::SetNextWindowPos(pos);
	ImGui::SetNextWindowSize(ImVec2{ width,50.0f });
	ImGui::Begin("controls", &state, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav);

	bool is_pause = m_video.get_pause_state();

	static char path[max_string_len]{ 0 };
	static ImVec2 button_size{ 80.0f,30.f };

	char z_select[max_string_len] = "select video file";
	char z_start[max_string_len] = "start";
	char z_pause[max_string_len] = "pause";
	char z_display[max_string_len] = "continue";
	char z_exit[max_string_len] = "exit";
	char z_schedule[max_string_len] = "schedule";
	char z_fps[max_string_len] = "   FPS : %2.2lf    ";
	if (m_is_english == false)
	{
		to_utf8("选择视频文件", z_select, max_string_len);
		to_utf8("播放", z_start, max_string_len);
		to_utf8("暂停", z_pause, max_string_len);
		to_utf8("继续", z_display, max_string_len);
		to_utf8("退出", z_exit, max_string_len);
		to_utf8("进度", z_schedule, max_string_len);
	}

	if (ImGui::Button(z_select, { 120.0f,30.0f }))
	{
		get_file_path("video file\0*.mp4;*.flv;*.ts;*.avi\0\0", path, max_string_len);
		int size = strlen(path);
		if (size)
		{
			m_video.set_video_path(path);
			m_video.get_per_video_frame(path);
		}
	}
	ImGui::SameLine();

	if (ImGui::Button(z_start, button_size)) m_video.start();
	ImGui::SameLine();

	if (is_pause == false) { if (ImGui::Button(z_pause, button_size)) m_video.pause(); }
	else if (ImGui::Button(z_display, button_size)) m_video.restart();
	ImGui::SameLine();

	if (ImGui::Button(z_exit, button_size)) m_video.close();
	ImGui::SameLine();

	float value = m_video.get_finish_rate() * 100.0f;
	if (ImGui::SliderFloat(z_schedule, &value, 0.0f, 100.0f, "%2.1f"))
		m_video.set_frame_index(value / 100.0f);
	ImGui::SameLine();

	ImGui::TextColored(ImVec4{ 1.0f,0,0,1.0f }, z_fps, m_video.get_display_fps());

	ImGui::End();
}

void gui::imgui_region_manager() noexcept
{
	if (m_region_manager == false) return;

	char z_title[max_string_len] = "region manager";
	if (m_is_english == false)
	{
		to_utf8("区域管理", z_title, max_string_len);
	}

	ImGui::SetNextWindowPos(ImVec2{ 0,0 });
	ImGui::SetNextWindowSize(ImVec2{ 300.0f,300.0f }, ImGuiCond_FirstUseEver);
	if (ImGui::IsKeyPressed(112)) ImGui::SetNextWindowSize(ImVec2{ 300.0f,300.0f });
	ImGui::Begin(z_title, &m_region_manager, ImGuiWindowFlags_NoMove);

	imgui_select_video(false);
	static struct frame_handle last_data = {};
	struct frame_handle* data = m_video.get_video_frame();
	if (data)
	{
		if (last_data.frame.empty() == false) last_data.frame.release();
		data->frame.copyTo(last_data.frame);
		data->frame.release();
		delete data;
	}

	//按键space
	if (ImGui::IsKeyPressed(32))
		if (last_data.frame.empty() == false)
			last_data.frame.release();

	update_texture(&last_data);
	if (m_IDirect3DTexture9) ImGui::Image(m_IDirect3DTexture9, ImGui::GetContentRegionAvail());

	{
		if (last_data.frame.empty())
		{
			char z_text1[max_string_len] = "introduction : ";
			char z_text2[max_string_len] = "press [VK_F1] to reset the window size to 300x300";
			char z_text3[max_string_len] = "press [VK_DELETE] to delete the last locale";
			char z_text4[max_string_len] = "press [VK_SPACE] to delete the video frame setting";
			char z_text5[max_string_len] = "click the left mouse button to set the area information";
			char z_text6[max_string_len] = "click the right mouse button to set the video frame";
			char z_text7[max_string_len] = "press [VK_NUMBER0] to set the bus lane area";
			char z_text8[max_string_len] = "press [VK_NUMBER1] to set the zebra crossing area";
			if (m_is_english == false)
			{
				to_utf8("介绍 : ", z_text1, max_string_len);
				to_utf8("按[VK_F1]重置窗口大小为300x300", z_text2, max_string_len);
				to_utf8("按[VK_DELETE]删除上一个区域设置", z_text3, max_string_len);
				to_utf8("按[VK_SPACE]删除视频帧设置", z_text4, max_string_len);
				to_utf8("单击鼠标左键设置区域信息", z_text5, max_string_len);
				to_utf8("单击鼠标右键设置视频帧", z_text6, max_string_len);
				to_utf8("按[VK_NUMBER0]设置公交车道区域", z_text7, max_string_len);
				to_utf8("按[VK_NUMBER1]设置斑马线区域", z_text8, max_string_len);
			}

			ImGui::Text(z_text1);
			ImGui::Text(z_text2);
			ImGui::Text(z_text3);
			ImGui::Text(z_text4);
			ImGui::Text(z_text5);
			ImGui::Text(z_text6);
			ImGui::Text(z_text7);
			ImGui::Text(z_text8);
		}
	}

	static enum region_type reg_type = region_bus;
	if (GetAsyncKeyState(VK_NUMPAD0) & 0x8000) reg_type = region_bus;
	if (GetAsyncKeyState(VK_NUMPAD1) & 0x8000) reg_type = region_zebra_cross;

	//获取绘制指针
	ImDrawList* draw_list = ImGui::GetWindowDrawList();

	//x减去8.0f  y减去25.0f
	ImVec2 win_size = ImGui::GetWindowSize();
	ImVec2 mouse_size = ImGui::GetIO().MousePos;
	ImVec2 user_size = ImGui::GetContentRegionAvail();
	ImVec2 cursor_size = ImGui::GetCursorScreenPos();

	//ImGui::Text(u8"窗口大小 : %f  %f ", win_size.x, win_size.y);
	//ImGui::Text(u8"鼠标位置 : %f  %f ", mouse_size.x, mouse_size.y);
	//ImGui::Text(u8"可用大小 : %f  %f ", user_size.x, user_size.y);
	//ImGui::Text(u8"鼠标屏幕 : %f  %f ", cursor_size.x, cursor_size.y);

	static bool first_down = false;
	static ImVec2 begin_pos{ 0,0 };

	//鼠标左键按下
	if (ImGui::IsMouseClicked(0))
	{
		if (mouse_size.x < win_size.x  && mouse_size.y < win_size.y)
		{
			if (first_down == false)
			{
				first_down = true;
				begin_pos = mouse_size;
			}
			else
			{
				first_down = false;

				float x = abs(begin_pos.x - mouse_size.x);
				float y = abs(begin_pos.y - mouse_size.y);
				if (x > 30.0f || y > 30.0f)
				{
					struct region_info temp;
					temp.rect.w = win_size.x;
					temp.rect.h = win_size.y;
					temp.type = reg_type;
					if (reg_type == region_bus) temp.color = ImColor(ImVec4(1.0f, 0.0f, 0.4f, 1.0f));
					if (reg_type == region_zebra_cross) temp.color = ImColor(ImVec4(0, 1.0f, 0.4f, 1.0f));
					temp.rect.left = begin_pos.x;
					temp.rect.top = begin_pos.y;
					temp.rect.right = mouse_size.x;
					temp.rect.down = mouse_size.y;
					m_video.push_region_back(temp);
				}
			}
		}
	}

	//按键delete
	if (ImGui::IsKeyPressed(8))
	{
		first_down = false;
		m_video.pop_region_back();
	}

	const ImU32 color = ImColor(ImVec4(1.0f, 1.0f, 0.4f, 1.0f));
	if (first_down)
		draw_list->AddRect(begin_pos, mouse_size, color, 0.0f, 0, 0);

	auto region_list = m_video.get_region_list();
	for (auto& it : region_list)
		draw_list->AddRect({ (float)it.rect.left,(float)it.rect.top }, { (float)it.rect.right,(float)it.rect.down }, it.color, 0.0f, 0, 0);

	ImGui::End();
}

void gui::imgui_people_statistics() noexcept
{
	if (m_people_statistics == false) return;

	char z_title[max_string_len] = "people statistics";
	char z_current[max_string_len] = "current people statistics %d ";
	char z_max[max_string_len] = "max people statistics %d ";
	char z_data[max_string_len] = "statistics data";
	char z_save[max_string_len] = "save to file";
	char z_clear[max_string_len] = "clear statistics data";
	if (m_is_english == false)
	{
		to_utf8("人流量统计", z_title, max_string_len);
		to_utf8("当前人流量 %d ", z_current, max_string_len);
		to_utf8("最大人流量 %d ", z_max, max_string_len);
		to_utf8("统计数据", z_data, max_string_len);
		to_utf8("保存为文件", z_save, max_string_len);
		to_utf8("清除统计数据", z_clear, max_string_len);
	}

	ImGui::SetNextWindowSize(ImVec2{ 550,300 });
	ImGui::Begin(z_title, &m_people_statistics);

	struct calc_statistics_info* people_info = m_video.get_people_info_point();
	if (people_info->enable)
	{
		ImGui::Text(z_current, people_info->current_val);
		ImGui::SameLine();
		ImGui::Text(z_max, people_info->max_val);

		static float max_height = 0;
		float *p_data = nullptr;
		int size = people_info->val_list.size();
		if (size <= 0)
		{
			p_data = new float[1];
			p_data[0] = 0.0f;
		}
		else
		{
			p_data = new float[size];
			for (int i = 0; i < size; i++)
			{
				p_data[i] = people_info->val_list[i];
				if (p_data[i] > max_height) max_height = p_data[i];
			}
		}

		ImGui::PlotHistogram(z_data, p_data, size, 0, nullptr, 0.0f, max_height, ImVec2(500, 100));
		delete[] p_data;

		if (ImGui::Button(z_save)) {}
		ImGui::SameLine();
		if (ImGui::Button(z_clear)) people_info->clear();

		people_info->entry_image();
		for (auto it = people_info->val_images.rbegin(); it != people_info->val_images.rend(); it++)
		{
			IDirect3DTexture9* res = get_image_texture(&(*it));
			if (res)
			{
				ImGui::Image(res, { 100,100 });
				ImGui::SameLine();
			}
		}
		people_info->leave_image();
	}

	ImGui::End();
}

void gui::imgui_car_statistics() noexcept
{
	if (m_car_statistics == false) return;

	char z_title[max_string_len] = "car statistics";
	char z_current[max_string_len] = "current people statistics %d ";
	char z_max[max_string_len] = "max people statistics %d ";
	char z_data[max_string_len] = "statistics data";
	char z_save[max_string_len] = "save to file";
	char z_clear[max_string_len] = "clear data";
	if (m_is_english == false)
	{
		to_utf8("车辆统计", z_title, max_string_len);
		to_utf8("当前车流量 %d ", z_current, max_string_len);
		to_utf8("最大车流量 %d ", z_max, max_string_len);
		to_utf8("统计数据", z_data, max_string_len);
		to_utf8("保存到文件", z_save, max_string_len);
		to_utf8("清除统计数据", z_clear, max_string_len);
	}

	ImGui::SetNextWindowSize(ImVec2{ 550,300 });
	ImGui::Begin(z_title, &m_car_statistics);

	struct calc_statistics_info* car_info = m_video.get_car_info_point();
	if (car_info->enable)
	{
		ImGui::Text(z_current, car_info->current_val);
		ImGui::SameLine();
		ImGui::Text(z_max, car_info->max_val);

		static float max_height = 0;
		float *p_data = nullptr;
		int size = car_info->val_list.size();
		if (size <= 0)
		{
			p_data = new float[1];
			p_data[0] = 0.0f;
		}
		else
		{
			p_data = new float[size];
			for (int i = 0; i < size; i++)
			{
				p_data[i] = car_info->val_list[i];
				if (p_data[i] > max_height) max_height = p_data[i];
			}
		}

		ImGui::PlotHistogram(z_data, p_data, size, 0, nullptr, 0.0f, max_height, ImVec2(500, 100));
		delete[] p_data;

		if (ImGui::Button(z_save)) {}
		ImGui::SameLine();
		if (ImGui::Button(z_clear)) car_info->clear();

		car_info->entry_image();
		for (auto it = car_info->val_images.rbegin(); it != car_info->val_images.rend(); it++)
		{
			IDirect3DTexture9* res = get_image_texture(&(*it));
			if (res)
			{
				ImGui::Image(res, { 100,100 });
				ImGui::SameLine();
			}
		}
		car_info->leave_image();
	}

	ImGui::End();
}

void gui::imgui_window_meun() noexcept
{
	if (ImGui::BeginMainMenuBar())
	{
		imgui_model_window();
		imgui_features_window();
		imgui_win_window();
		imgui_language_window();
		ImGui::EndMainMenuBar();
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
		to_utf8("车牌识别模型", z_recognition_model, max_string_len);
		to_utf8("加载模型", z_Load_model, max_string_len);
		to_utf8("卸载模型", z_Unload_model, max_string_len);
	}

	bool is_load_detect = m_video.get_detect_model()->get_model_loader();
	bool is_load_recognition = m_video.get_recognition_model()->is_loaded();

	if (ImGui::BeginMenu(z_model))
	{
		if (ImGui::BeginMenu(z_detect_model))
		{
			if (ImGui::MenuItem(z_Load_model, nullptr, false, !is_load_detect))
			{
				char buffer[max_string_len]{ 0 };
				get_file_path("Model \0*.model\0\0", buffer, max_string_len);

				int len = strlen(buffer);
				object_detect* model = m_video.get_detect_model();
				if (len && model->set_model_path(buffer))
				{
					if (model->load_model() == false) {}
				}
				else check_warning(false, "设置模型文件失败");
			}
			if (ImGui::MenuItem(z_Unload_model, nullptr, false, is_load_detect))
			{
				m_video.get_detect_model()->unload_model();
			}
			ImGui::EndMenu();
		}

		if (ImGui::BeginMenu(z_recognition_model))
		{
			if (ImGui::MenuItem(z_Load_model, nullptr, false, !is_load_recognition))
			{
				char buffer[max_string_len]{ 0 };
				get_file_path("Model \0*.model\0\0", buffer, max_string_len);

				int len = strlen(buffer);
				object_recognition* recognition = m_video.get_recognition_model();
				if (len && recognition->set_model_path(buffer))
				{
					recognition->load_model();
				}
			}
			if (ImGui::MenuItem(z_Unload_model, nullptr, false, is_load_recognition))
			{
				m_video.get_recognition_model()->unload_model();
			}
			ImGui::EndMenu();
		}
		ImGui::EndMenu();
	}
}

void gui::imgui_features_window() noexcept
{
	char z_title[max_string_len] = "feature";
	char z_calc_people[max_string_len] = "people flow detection";
	char z_calc_car[max_string_len] = "vehicle flow detection";
	char z_occupy_bus[max_string_len] = "occupy the bus lane";

	if (m_is_english == false)
	{
		to_utf8("功能", z_title, max_string_len);
		to_utf8("统计人流量", z_calc_people, max_string_len);
		to_utf8("统计车流量", z_calc_car, max_string_len);
		to_utf8("占用公交车道", z_occupy_bus, max_string_len);
	}

	struct calc_statistics_info* people_info = m_video.get_people_info_point();
	struct calc_statistics_info* car_info = m_video.get_car_info_point();

	if (ImGui::BeginMenu(z_title))
	{
		static bool enabled = false;
		ImGui::MenuItem(z_calc_people, nullptr, &people_info->enable);
		ImGui::MenuItem(z_calc_car, nullptr, &car_info->enable);
		ImGui::MenuItem(z_occupy_bus, nullptr, &enabled);
		ImGui::EndMenu();
	}
}

void gui::imgui_win_window() noexcept
{
	char z_title[max_string_len] = "Windows";
	char z_region[max_string_len] = "region manager";
	char z_people[max_string_len] = "people statistics";
	char z_car[max_string_len] = "car statistics";
	if (m_is_english == false)
	{
		to_utf8("窗口", z_title, max_string_len);
		to_utf8("区域管理", z_region, max_string_len);
		to_utf8("人流量统计", z_people, max_string_len);
		to_utf8("车流量统计", z_car, max_string_len);
	}

	if (ImGui::BeginMenu(z_title))
	{
		if (ImGui::MenuItem(z_region)) m_region_manager = true;
		if (ImGui::MenuItem(z_people)) m_people_statistics = true;
		if (ImGui::MenuItem(z_car)) m_car_statistics = true;
		ImGui::EndMenu();
	}
}

void gui::imgui_language_window() noexcept
{
	char z_title[max_string_len] = "Language";
	char z_english[max_string_len] = "English";
	char z_chinese[max_string_len] = "Chinese";
	if (m_is_english == false)
	{
		to_utf8("语言", z_title, max_string_len);
		to_utf8("英语", z_english, max_string_len);
		to_utf8("中文", z_chinese, max_string_len);
	}

	if (ImGui::BeginMenu(z_title))
	{
		if (ImGui::MenuItem(z_english))  set_english_display(true);
		if (ImGui::MenuItem(z_chinese)) set_english_display(false);
		ImGui::EndMenu();
	}
}

gui::gui() noexcept
{
	m_is_english = false;

	m_region_manager = false;
	m_people_statistics = false;
	m_car_statistics = false;

	m_IDirect3D9 = nullptr;
	m_IDirect3DDevice9 = nullptr;
	m_IDirect3DTexture9 = nullptr;

	m_D3DPRESENT_PARAMETERS = { 0 };
}

gui::~gui() noexcept
{
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
		10, 10, 1200, 700, 0, 0, GetModuleHandleA(nullptr), 0);
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