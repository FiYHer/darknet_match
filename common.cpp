#include "common.h"

void check_serious_error(bool state, const char* show_str /*= ""*/)
{
	if (!state)
	{
		char buffer[1024];
		sprintf(buffer, "发生严重错误!\t 错误提示:%s\t 文件:%s\t 行数:%d\t \n", show_str, __FILE__, __LINE__);
		MessageBoxA(NULL, buffer, NULL, NULL);
		exit(-1);
	}
}

void show_window_tip(const char* str)
{
	MessageBoxA(NULL, str, NULL, NULL);
}

int get_gpu_count()
{
	int gpu_count = 0;
	check_serious_error(cudaGetDeviceCount(&gpu_count) == cudaSuccess, "获取显卡数量失败");
	check_serious_error(gpu_count, "没有发现电脑显卡");
	return gpu_count;
}

cudaDeviceProp* get_gpu_infomation(int gpu_count)
{
	//获取显卡相关信息
	cudaDeviceProp* gpu_info = new cudaDeviceProp[gpu_count];
	check_serious_error(gpu_info, "申请显卡相关内存失败");
	for (int i = 0; i < gpu_count; i++) check_serious_error(cudaGetDeviceProperties(&gpu_info[i], i) == cudaSuccess, "获取显卡信息失败");
	return gpu_info;
}

int get_os_type()
{
	typedef void(__stdcall *get_os)(DWORD*, DWORD*, DWORD*);
	get_os func = (get_os)GetProcAddress(LoadLibraryA("ntdll.dll"), "RtlGetNtVersionNumbers");
	if (func)
	{
		DWORD nativeMajor = 0, nativeMinor = 0, dwBuildNumber = 0;
		func(&nativeMajor, &nativeMinor, &dwBuildNumber);
		if (nativeMajor == 6 && nativeMinor == 1) return 7;
		if (nativeMajor == 6 && nativeMinor == 3) return 8;
		if (nativeMajor == 10) return 10;
	}
	return -1;
}

int get_cpu_kernel()
{
	SYSTEM_INFO info;
	GetSystemInfo(&info);
	return info.dwNumberOfProcessors;
}

int get_physical_memory()
{
	MEMORYSTATUS memory;
	GlobalMemoryStatus(&memory);
	return memory.dwTotalPhys / 1024 / 1024 / 1024;
}

bool select_type_file(const char* type_file, char* return_str,int return_str_size)
{
	OPENFILENAMEA open_file{0};
	open_file.lStructSize = sizeof(open_file);
	open_file.hwndOwner = NULL;
	open_file.lpstrFilter = type_file;
	open_file.nFilterIndex = 1;
	open_file.lpstrFile = return_str;
	open_file.nMaxFile = return_str_size;
	open_file.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_HIDEREADONLY;
	return GetOpenFileName(&open_file);
}

void read_classes_name(std::vector<std::string>& return_data, const char* path)
{
	std::fstream file(path, std::fstream::in);
	if (!file.is_open())
	{
		show_window_tip("文件读取失败");
		return;
	}

	return_data.clear();
	int index = 0;
	std::string line_data;
	while (getline(file, line_data))
	{
		if (line_data.size())
		{
			char temp[default_char_size];
			sprintf(temp, "Index : %d    Name : %s", index++, line_data.c_str());
			return_data.push_back(std::move(temp));
		}
	}

	file.close();
}

bool initialize_net(const char* names_file, const char* cfg_file, const char* weights_file)
{
	if (g_global_set.net_set.initizlie)
	{
		show_window_tip("网络不需要再次初始化");
		return true;
	}

	//判断names文件是否存在
	if (!names_file || !strstr(names_file, ".names") || access(names_file, 0) == -1)
	{
		show_window_tip("names文件不存在");
		return false;
	}

	//判断cfg文件是否存在
	if (!cfg_file || !strstr(cfg_file, ".cfg") || access(cfg_file, 0) == -1)
	{
		show_window_tip("cfg文件不存在");
		return false;
	}

	//判断weights文件是否存在
	if (!weights_file || !strstr(weights_file, ".weights") || access(weights_file, 0) == -1)
	{
		show_window_tip("weights文件不存在");
		return false;
	}

	//设置显卡工作
	cuda_set_device(get_gpu_count() - 1);
	CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));

	//读取标签数据
	int classes_number;
	g_global_set.net_set.classes_name = get_labels_custom((char*)names_file, &classes_number);

	//读取网络数据
	g_global_set.net_set.match_net = parse_network_cfg_custom((char*)cfg_file, 1, 1);

	//加载权重文件
	load_weights(&g_global_set.net_set.match_net, (char*)weights_file);

	//融合卷积
	fuse_conv_batchnorm(g_global_set.net_set.match_net);

	//计算二进制权重
	calculate_binary_weights(g_global_set.net_set.match_net);

	//检测标签
	check_serious_error(g_global_set.net_set.match_net.layers[g_global_set.net_set.match_net.n - 1].classes == classes_number, "和yolo层标签数不符");
	g_global_set.net_set.classes = classes_number;

	//设置状态
	g_global_set.net_set.initizlie = true;

	//返回结果
	return true;
}

void clear_net()
{
	if (g_global_set.net_set.classes_name) free_ptrs((void**)g_global_set.net_set.classes_name, g_global_set.net_set.classes);
	free_network(g_global_set.net_set.match_net);
	g_global_set.net_set.initizlie = false;
}

void read_picture_data(const char* target, image& picture_data, cv::Mat& opencv_data, cv::Mat& rgb_data)
{
	//记载图片数据
	opencv_data = cv::imread(target);
	if (opencv_data.empty()) return;
	
	//转化颜色通道
	if (opencv_data.channels() == 3) cv::cvtColor(opencv_data, rgb_data, cv::COLOR_RGB2BGR);
	else if (opencv_data.channels() == 4) cv::cvtColor(opencv_data, rgb_data, cv::COLOR_RGBA2BGRA);
	else opencv_data.copyTo(rgb_data);

	//获取图片信息
	int width = rgb_data.cols, height = rgb_data.rows, channel = rgb_data.channels();

	//转化为image结构顺便均值化
	mat_translate_image(rgb_data, picture_data);
}

void mat_translate_image(const cv::Mat& opencv_data, image& image_data)
{
	int width = opencv_data.cols;
	int height = opencv_data.rows;
	int channel = opencv_data.channels();

	image_data = make_image(width, height, channel);
	int step = opencv_data.step;
	for (int y = 0; y < height; ++y)
	{
		for (int k = 0; k < channel; ++k)
		{
			for (int x = 0; x < width; ++x)
			{
				image_data.data[k*width*height + y * width + x] = opencv_data.data[y*step + x * channel + k] / 255.0f;
			}
		}
	}
}

void analyse_picture(const char* target, picture_detect_info& detect_info, bool show)
{
	//加载图片
	image original_data;
	cv::Mat opencv_data, rgb_data;
	read_picture_data(target, original_data, opencv_data, rgb_data);
	if (opencv_data.empty())
	{
		show_window_tip("图片打开失败");
		return;
	}

	//将图片数据进行缩放
	image resize_data = resize_image(original_data, g_global_set.net_set.match_net.w, g_global_set.net_set.match_net.h);

	//获取开始时间
	double this_time = get_time_point();

	//开始预测
	network_predict(g_global_set.net_set.match_net, resize_data.data);

	//计算预测需要的时间
	detect_info.detect_time = ((double)get_time_point() - this_time) / 1000;

	//获取方框数量
	int box_number;
	detection* detection_data = get_network_boxes(&g_global_set.net_set.match_net,
		original_data.w, original_data.h, detect_info.thresh, detect_info.hier_thresh, 0, 1, &box_number, 0);

	//非极大值抑制
	do_nms_sort(detection_data, box_number, g_global_set.net_set.classes, detect_info.nms);

	//获取有效的方框数量
	int useble_box = 0;
	detection_with_class* detector_data = get_actual_detections(detection_data,
		box_number, detect_info.thresh, &useble_box, g_global_set.net_set.classes_name);

	//对每一个对象
	for (int i = 0; i < useble_box; i++)
	{
		int index = detector_data[i].best_class;//获取类型
		int confid = detector_data[i].det.prob[detector_data[i].best_class] * 100;//获取置信度

		char format[1024];
		sprintf(format, "%s %d", g_global_set.net_set.classes_name[index], confid);
		draw_boxs_and_classes(opencv_data, detector_data[i].det.bbox, format);
		//计算位置信息
		//object_info temp_object;
		//get_object_rect(original_data.w, original_data.h, detector_data[i].det.bbox, temp_object);

		//绘制方框
		//draw_object_rect(opencv_data, temp_object.left, temp_object.top, temp_object.right, temp_object.down);

		//绘制字体
		//cv::putText(opencv_data, cv::format("%s %d", g_global_set.net_set.classes_name[index], confid), cv::Point(temp_object.left, temp_object.top), cv::FONT_HERSHEY_COMPLEX, .50f, cv::Scalar(0, 0, 255), 0);
	}

	//释放内存
	free_detections(detection_data, box_number);
	free_image(original_data);
	free_image(resize_data);
	free(detector_data);

	//opencv界面显示
	if (show)
	{
		cv::imshow(target, opencv_data);
		cv::waitKey(1);
	}

	//更新置纹理数据
	update_picture_texture(opencv_data);

	//清空数据
	opencv_data.release();
	rgb_data.release();
}

void draw_boxs_and_classes(cv::Mat& picture_data, box box_info, const char* name)
{
	//计算方框位置
	if (std::isnan(box_info.w) || std::isinf(box_info.w)) box_info.w = 0.5;
	if (std::isnan(box_info.h) || std::isinf(box_info.h)) box_info.h = 0.5;
	if (std::isnan(box_info.x) || std::isinf(box_info.x)) box_info.x = 0.5;
	if (std::isnan(box_info.y) || std::isinf(box_info.y)) box_info.y = 0.5;

	box_info.w = (box_info.w < 1) ? box_info.w : 1;
	box_info.h = (box_info.h < 1) ? box_info.h : 1;
	box_info.x = (box_info.x < 1) ? box_info.x : 1;
	box_info.y = (box_info.y < 1) ? box_info.y : 1;

	int left = (box_info.x - box_info.w / 2.)*picture_data.cols;
	int right = (box_info.x + box_info.w / 2.)*picture_data.cols;
	int top = (box_info.y - box_info.h / 2.)*picture_data.rows;
	int bot = (box_info.y + box_info.h / 2.)*picture_data.rows;

	if (left < 0) left = 0;
	if (right > picture_data.cols - 1) right = picture_data.cols - 1;
	if (top < 0) top = 0;
	if (bot > picture_data.rows - 1) bot = picture_data.rows - 1;

	//计算字体大小
	float font_size = picture_data.rows / 1000.0f;
	cv::Size text_size = cv::getTextSize(name, cv::FONT_HERSHEY_COMPLEX_SMALL, font_size, 1, 0);

	//方框颜色
	float box_rgb[3]{ g_global_set.color_set.box_rgb[0] * 255.0f, g_global_set.color_set.box_rgb[1] * 255.0f,g_global_set.color_set.box_rgb[2] * 255.0f };
	
	//字体颜色
	float font_rgb[3]{ g_global_set.color_set.font_rgb[0] * 255.0f,g_global_set.color_set.font_rgb[1] * 255.0f ,g_global_set.color_set.font_rgb[2] * 255.0f };

	//线条粗细
	float thickness = g_global_set.color_set.thickness;

	//绘制人物方框
	cv::rectangle(picture_data, cv::Point(left, top), cv::Point(right, bot), cv::Scalar(box_rgb[2], box_rgb[1], box_rgb[0]), thickness, 8, 0);

	//计算字体方框位置
	cv::Point font_left{ left,top };
	cv::Point font_right{ right,top + text_size.height * 2 };
	if (font_right.y > picture_data.rows) font_right.y = picture_data.rows - 1;

	//绘制字体方框
	cv::rectangle(picture_data, font_left, font_right, cv::Scalar(font_rgb[2], font_rgb[1], font_rgb[0]), thickness, 8, 0);
	cv::rectangle(picture_data, font_left, font_right, cv::Scalar(font_rgb[2], font_rgb[1], font_rgb[0]), cv::FILLED, 8, 0);

	//绘制字体
	cv::Point pos{ font_left.x, font_left.y + text_size.height };
	if (pos.x + text_size.width > picture_data.cols) pos.x = picture_data.cols - text_size.width - 1;
	if (pos.y + text_size.height > picture_data.rows) pos.y = picture_data.rows - text_size.height - 1;
	cv::putText(picture_data, name, pos, cv::FONT_HERSHEY_COMPLEX_SMALL, font_size, cv::Scalar(255 - font_rgb[2], 255 - font_rgb[1], 255 - font_rgb[0]), 2 * font_size, cv::LINE_AA);
}

void get_object_rect(int width, int height, box& box_pos, object_info& object)
{
	int left = (box_pos.x - box_pos.w / 2.)*width;
	int right = (box_pos.x + box_pos.w / 2.)*width;
	int top = (box_pos.y - box_pos.h / 2.)*height;
	int down = (box_pos.y + box_pos.h / 2.)*height;

	if (left < 0) left = 1;
	if (right > width - 1) right = width - 1;
	if (top < 0) top = 1;
	if (down > height - 1) down = height - 1;

	object.left = left;
	object.top = top;
	object.right = right;
	object.down = down;
}

void draw_object_rect(cv::Mat& buffer, int left, int top, int right, int down)
{
	int size = 3;
	int width = buffer.cols;
	int height = buffer.rows;
	int channel = buffer.channels();
	static int r = rand() % 255, g = rand() % 255, b = rand() % 255;

	for (int k = 0; k < size; k++)
	{
		for (int i = left; i < right; i++)
		{
			buffer.data[((top + k) * width + i) * channel + 0] = r;
			buffer.data[((top + k) * width + i) * channel + 1] = g;
			buffer.data[((top + k) * width + i) * channel + 2] = b;

			buffer.data[((down - k) * width + i) * channel + 0] = r;
			buffer.data[((down - k) * width + i) * channel + 1] = g;
			buffer.data[((down - k) * width + i) * channel + 2] = b;
		}
	}

	for (int k = 0; k < size; k++)
	{
		for (int i = top; i < down; i++)
		{
			buffer.data[(i * width + (left + k)) * channel + 0] = r;
			buffer.data[(i * width + (left + k)) * channel + 1] = g;
			buffer.data[(i * width + (left + k)) * channel + 2] = b;

			buffer.data[(i * width + (right - k)) * channel + 0] = r;
			buffer.data[(i * width + (right - k)) * channel + 1] = g;
			buffer.data[(i * width + (right - k)) * channel + 2] = b;
		}
	}
}

std::string string_to_utf8(const char* str)
{
	int nwLen = MultiByteToWideChar(CP_ACP, 0, str, -1, NULL, 0);
	wchar_t* pwBuf = new wchar_t[nwLen + 1];
	memset(pwBuf, 0, nwLen * 2 + 2);
	MultiByteToWideChar(CP_ACP, 0, str, strlen(str), pwBuf, nwLen);

	int nLen = WideCharToMultiByte(CP_UTF8, 0, pwBuf, -1, NULL, NULL, NULL, NULL);
	char* pBuf = new char[nLen + 1];
	memset(pBuf, 0, nLen + 1);
	WideCharToMultiByte(CP_UTF8, 0, pwBuf, nwLen, pBuf, nLen, NULL, NULL);

	std::string ret = pBuf;
	delete[]pwBuf;
	delete[]pBuf;

	return ret;
}

void update_picture_texture(cv::Mat& opencv_data)
{
	//获取图片信息
	int width = opencv_data.cols, height = opencv_data.rows, channel = opencv_data.channels();
	g_global_set.width = width;
	g_global_set.height = height;
	g_global_set.channel = channel + 1;

	//清空以前的图片数据
	if (g_global_set.picture_data)
	{
		delete[] g_global_set.picture_data;
		g_global_set.picture_data = nullptr;
	}

	//申请新的空间保存图片数据,为什么要通道加1？因为是RGBA格式
	g_global_set.picture_data = new unsigned char[width * height * (channel + 1)];
	check_serious_error(g_global_set.picture_data, "申请图片空间失败");

	//图片格式转化
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			g_global_set.picture_data[(i * width + j) * (channel + 1) + 0] = opencv_data.data[(i * width + j) * channel + 0];
			g_global_set.picture_data[(i * width + j) * (channel + 1) + 1] = opencv_data.data[(i * width + j) * channel + 1];
			g_global_set.picture_data[(i * width + j) * (channel + 1) + 2] = opencv_data.data[(i * width + j) * channel + 2];
			g_global_set.picture_data[(i * width + j) * (channel + 1) + 3] = 0xff;
		}
	}
}

unsigned __stdcall  analyse_video(void* prt)
{
	//转化
	video_control* control_ptr = (video_control*)prt;
	control_ptr->leave = false;

	//打开视频文件
	video_handle_info video_info;
	video_info.cap.open(control_ptr->video_path);
	if (!video_info.cap.isOpened())
	{
		show_window_tip("打开视频文件失败");
		control_ptr->leave = true;
		return 0;
	}

	//初始化相关信息
	video_info.initialize();

	//延迟设置
	video_info.show_delay = &control_ptr->show_delay;
	video_info.read_delay = &control_ptr->read_delay;
	video_info.detect_delay = &control_ptr->detect_delay;

	//创建线程
	int detect_threas = control_ptr->detect_count;
	HANDLE read_handle;
	HANDLE* detect_handle = new HANDLE[detect_threas];
	check_serious_error(detect_handle, "申请线程句柄内存失败");

	//创建读取线程
	read_handle = (HANDLE)_beginthreadex(NULL, 0, read_frame_proc, &video_info, 0, NULL);
	check_serious_error(read_handle, "创建读取视频帧线程失败");

	//创建检测线程
	for (int i = 0; i < detect_threas; i++)
	{
		detect_handle[i] = (HANDLE)_beginthreadex(NULL, 0, prediction_frame_proc, &video_info, 0, NULL);
		check_serious_error(detect_handle[i], "创建检测视频帧线程失败");
	}

	double fps = 0, before = 0;

	//循环显示视频帧
	while (!control_ptr->leave && video_info.detect_frame)
	{
		//视频帧指针
		video_frame_info* video_ptr = nullptr;

		//获取控制权
		video_info.entry();

		//如果有视频帧
		if (video_info.detect_datas.size())
		{
			//我们拿取第一个视频帧
			video_ptr = video_info.detect_datas[0];
			
			//能显示就显示
			if (video_ptr->display) video_info.detect_datas.erase(video_info.detect_datas.begin());
			else video_ptr = nullptr;//不能显示就什么都不做
		}

		//释放控制权
		video_info.leave();

		//如果有结果了
		if (video_ptr)
		{
			double after = get_time_point();    // more accurate time measurements
			double curr = 1000000. / (after - before);
			fps = fps * 0.9 + curr * 0.1;
			before = after;
			static char fps_char[default_char_size];
			sprintf(fps_char, "fps is : %d", (int)fps);
			cv::putText(video_ptr->original_frame, fps_char, cv::Point(50, 50), cv::FONT_HERSHEY_COMPLEX, 2, cv::Scalar(0, 0, 255));

			//显示结果
			cv::imshow(control_ptr->video_path, video_ptr->original_frame);
		}

		//显示
		cv::waitKey(*video_info.show_delay);

		//释放视频帧内存
		if (video_ptr)
		{
			video_ptr->original_frame.release();
			delete video_ptr;
		}
	}

	//关闭窗口
	cv::destroyWindow(control_ptr->video_path);

	//等待线程的全部退出
	video_info.break_state = true;
	video_info.read_frame = false;
	video_info.detect_frame = false;
	WaitForSingleObject(read_handle, 1000 * 3);
	WaitForMultipleObjects(detect_threas, detect_handle, TRUE, 1000 * 3);

	//释放相关内存
	video_info.clear();

	//设置退出标志
	control_ptr->leave = true;
	return 0;
}

unsigned __stdcall read_frame_proc(void* prt)
{
	//获取
	video_handle_info* video_info = (video_handle_info*)prt;

	//开始循环读取视频帧
	while (video_info->read_frame && !video_info->break_state)
	{
		//防止读取太快检测不过来而占用太多内存
		if (video_info->max_frame_count > video_info->detect_datas.size())
		{
			//申请内存
			video_frame_info* video_ptr = new video_frame_info;
			check_serious_error(video_ptr, "申请保存视频帧内存失败");
			video_ptr->detecting = false;//还没检测
			video_ptr->display = false;//不能显示

			//读取一帧视频
			if (!video_info->cap.read(video_ptr->original_frame)) break;

			//尝试取得控制权
			video_info->entry();

			//赋值指针
			video_info->detect_datas.push_back(video_ptr);

			//释放控制权
			video_info->leave();
		}

		//放过CPU
		Sleep(*video_info->read_delay);
	}

	//设置读取标志
	video_info->read_frame = false;
	return 0;
}

unsigned __stdcall prediction_frame_proc(void* prt)
{
	//转化
	video_handle_info* video_info = (video_handle_info*)prt;

	//网络的输入宽度和高度
	int input_width = g_global_set.net_set.match_net.w;
	int input_height = g_global_set.net_set.match_net.h;
	int input_channel = g_global_set.net_set.match_net.c;

	//类型数量
	int classes = g_global_set.net_set.classes;

	//开始检测视频帧
	while (video_info->detect_frame && video_info->read_frame && !video_info->break_state)
	{
		//
		video_frame_info* video_ptr = nullptr;

		//获取控制权
		video_info->entry();

		//找到一个还没开始检测的视频帧
		for (int i = 0; i < video_info->detect_datas.size(); i++)
		{
			if (video_info->detect_datas[i]->detecting == false)//找到没检测的视频帧
			{
				video_info->detect_datas[i]->detecting = true;//标记为正在检测
				video_ptr = video_info->detect_datas[i];//拿到地址
				break;//退出循环
			}
		}

		//释放控制权
		video_info->leave();

		//有视频帧让我们工作
		if (video_ptr)
		{
			//进行缩放
			cv::Mat picture_data = cv::Mat(input_height, input_width, CV_8UC(input_channel));
			cv::resize(video_ptr->original_frame, picture_data, picture_data.size(), 0, 0, cv::INTER_LINEAR);
			
			//颜色格式转换
			cv::Mat rgb_data;
			cv::cvtColor(picture_data, rgb_data, cv::COLOR_RGB2BGR);

			//将图像转换为image
			image original_data;
			mat_translate_image(rgb_data, original_data);

			//网络预测
			network_predict(g_global_set.net_set.match_net, original_data.data);

			//获取方框数量
			int box_count = 0;
			detection* detection_data = get_network_boxes(&g_global_set.net_set.match_net, input_width, input_height, .25f, .45f, 0, 1, &box_count, 0);

			//进行非极大值抑制
			do_nms_sort(detection_data, box_count, classes, .4f);

			//获取大于阈值的方框
			int useful_box = 0;
			detection_with_class* class_data = get_actual_detections(detection_data, box_count, .45f, &useful_box, g_global_set.net_set.classes_name);

			//操作每一个对象
			for (int i = 0; i < useful_box; i++)
			{
				//对象
				object_info object;

				//获取类别索引
				object.class_index = class_data[i].best_class;

				//获取置信度
				object.confidence = class_data[i].det.prob[object.class_index] * 100.0f;

				//获取方框位置
				get_object_rect(video_ptr->original_frame.cols, video_ptr->original_frame.rows, class_data[i].det.bbox, object);

				//绘制方框
				draw_object_rect(video_ptr->original_frame, object.left, object.top, object.right, object.down);
			}

			//能进行显示通知
			video_ptr->display = true;

			//释放图像数据
			free_image(original_data);
			picture_data.release();
			rgb_data.release();

			//释放内存
			free_detections(detection_data, box_count);
			free(class_data);
		}

		Sleep(*video_info->detect_delay);//暂停
	}

	//设置检测线程状态
	video_info->detect_frame = false;
	return 0;
}








std::vector<std::string> get_path_from_str(const char* str, const char* file_type)
{
	std::vector<std::string> res;

	char buffer[1024];
	sprintf(buffer, "%s\\%s", str, file_type);

	WIN32_FIND_DATAA find_data;
	HANDLE file_handle = FindFirstFileA(buffer, &find_data);
	assert(file_handle);

	do
	{
		res.push_back(find_data.cFileName);
	} while (FindNextFileA(file_handle, &find_data));

	return res;
}

//将图片转化为标签
void picture_to_label(const char* path, std::map<std::string, int>& class_names)
{
	//标签文件数量
	int names_size = 0;

	//读取标签名字
	char **names = get_labels_custom("h:\\test\\coco.names", &names_size); //get_labels(name_list);

	//读取网络配置
	network net = parse_network_cfg_custom("h:\\test\\yolov3.cfg", 1, 1); // set batch=1

	//加载权重文件
	load_weights(&net, "h:\\test\\yolov3.weights");

	//融合卷积
	fuse_conv_batchnorm(net);

	//计算二进制权重
	calculate_binary_weights(net);

	//获取图片文件
	std::vector<std::string> picture_list = get_path_from_str(path, "*.jpg");

	printf("图片总数量 %d \n", picture_list.size());

	//遍历每一张图片
	for (int i = 0; i < picture_list.size(); i++)
	{
		if (i && i % 100 == 0) printf("完成数量 : %d \n", i);

		//构建名称
		std::string jpg_name = path;
		jpg_name += "\\" + picture_list[i];

		//加载图像后转化大小
		image im = load_image((char*)jpg_name.c_str(), 0, 0, net.c);
		if(im.data == NULL) continue;
		image sized = resize_image(im, net.w, net.h);

		//网络预测
		network_predict(net, sized.data);

		//获取对象信息
		int nboxes = 0;
		detection *dets = get_network_boxes(&net, im.w, im.h, .25f, .5f, 0, 1, &nboxes, 0);

		//非极大值抑制
		do_nms_sort(dets, nboxes, net.layers[net.n - 1].classes, .4f);

		//获取有效方框
		int detections_num;
		detection_with_class* selected_detections = get_actual_detections(dets, nboxes, .25f, &detections_num, names);

		//有对象就创建文件
		if (detections_num)
		{
			//构建文件名字
			std::string file_name = jpg_name.substr(0, jpg_name.rfind('.'));
			file_name += ".txt";

			int useful_count = 0;

			//写入位置
			std::fstream file(file_name, std::fstream::out | std::fstream::trunc);
			if (file.is_open())
			{
				//对每一个对象
				for (int j = 0; j < detections_num; j++)
				{
					//获取类索引
					int class_index = selected_detections[j].best_class;

					//类型名称一样
					for (auto& it : class_names)
					{
						if (it.first == names[class_index])
						{
							//获取位置
							box b = selected_detections[j].det.bbox;

							//写入信息
							char format[1024];
							sprintf(format, "%d %f %f %f %f\n", it.second, b.x, b.y, b.w, b.h);
							file.write(format, strlen(format));

							//计数
							useful_count++;

							//退出当前循环
							break;
						}
					}
				}
			}

			//关闭文件
			file.close();

			//没有对象就删除文件
			if (useful_count == 0) DeleteFileA(file_name.c_str());
		}

		//释放内存
		free_detections(dets, nboxes);
		free_image(im);
		free_image(sized);
		free(selected_detections);
	}
}
