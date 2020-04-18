#include "common.h"

void check_serious_error(bool state, const char* show_str, const char* file_pos, int line_pos)
{
	if (!state)
	{
		char buffer[default_char_size];
		sprintf(buffer, "发生严重错误!\t 错误提示:%s\t 文件:%s\t 行数:%d\n", show_str, file_pos, line_pos);
		MessageBoxA(NULL, buffer, NULL, MB_ICONHAND | MB_OK);
		exit(-1);
	}
}

void show_window_tip(const char* str)
{
	MessageBoxA(NULL, str, "警告", MB_ICONWARNING | MB_OK);
}

int get_gpu_count()
{
	//当前为GPU版本的，如果没有显卡支持将无法进行计算
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
		if (nativeMajor == 6 && nativeMinor == 1) return 7;//Win7
		if (nativeMajor == 6 && nativeMinor == 3) return 8;//Win8
		if (nativeMajor == 10) return 10;//Win10
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
	OPENFILENAMEA open_file{ 0 };
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

bool initialize_object_detect_net(const char* names_file, const char* cfg_file, const char* weights_file)
{
	//防止二次初始化
	if (g_global_set.object_detect_net_set.initizlie)
	{
		show_window_tip("物体检测网络不需要再次初始化");
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

	//读取标签数据
	int classes_number;
	g_global_set.object_detect_net_set.classes_name = get_labels_custom((char*)names_file, &classes_number);

	//读取网络数据
	g_global_set.object_detect_net_set.this_net = parse_network_cfg_custom((char*)cfg_file, 1, 1);

	//加载权重文件
	load_weights(&g_global_set.object_detect_net_set.this_net, (char*)weights_file);

	//batch层融合进卷积层
	fuse_conv_batchnorm(g_global_set.object_detect_net_set.this_net);

	//计算二进制权重
	calculate_binary_weights(g_global_set.object_detect_net_set.this_net);

	//检测标签
	if (g_global_set.object_detect_net_set.this_net.layers[g_global_set.object_detect_net_set.this_net.n - 1].classes != classes_number)
	{
		show_window_tip("网络标签数与配置文件标签数不一致");
		clear_object_detect_net();
		return false;
	}

	//保存类别数量
	g_global_set.object_detect_net_set.classes = classes_number;

	//设置状态
	g_global_set.object_detect_net_set.initizlie = true;

	//返回结果
	return true;
}

bool initialize_car_id_identify_net(const char* names_file, const char* cfg_file, const char* weights_file)
{
	//防止二次初始化
	if (g_global_set.car_id_identify_net.initizlie)
	{
		show_window_tip("车牌识别网络不需要再次初始化");
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

	//加载网络
	g_global_set.car_id_identify_net.this_net = parse_network_cfg_custom((char*)cfg_file, 1, 0);

	//加载权重
	load_weights(&g_global_set.car_id_identify_net.this_net, (char*)weights_file);

	//设置批量为1
	set_batch_network(&g_global_set.car_id_identify_net.this_net, 1);

	//融合batch进卷积层
	fuse_conv_batchnorm(g_global_set.car_id_identify_net.this_net);

	//计算二进制权重
	calculate_binary_weights(g_global_set.car_id_identify_net.this_net);

	//获取标签数
	g_global_set.car_id_identify_net.classes_name = get_labels_custom((char*)names_file, &g_global_set.car_id_identify_net.classes);

	//设置状态
	g_global_set.car_id_identify_net.initizlie = true;

	return true;
}

void clear_object_detect_net()
{
	//不需要释放
	if (!g_global_set.object_detect_net_set.initizlie) return;

	//释放标签字符串
	if (g_global_set.object_detect_net_set.classes_name) free_ptrs((void**)g_global_set.object_detect_net_set.classes_name, g_global_set.object_detect_net_set.classes);
	
	//释放网络
	free_network(g_global_set.object_detect_net_set.this_net);

	//设置状态
	g_global_set.object_detect_net_set.initizlie = false;
}

void clear_car_id_identify_net()
{
	//不需要释放
	if (!g_global_set.car_id_identify_net.initizlie) return;

	//释放字符串
	if (g_global_set.car_id_identify_net.classes_name) free_ptrs((void**)g_global_set.car_id_identify_net.classes_name, g_global_set.car_id_identify_net.classes);

	//释放网络
	free_network(g_global_set.car_id_identify_net.this_net);

	//设置状态
	g_global_set.car_id_identify_net.initizlie = false;
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

void analyse_picture(const char* target, set_detect_info& detect_info, bool show)
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
	image resize_data = resize_image(original_data, g_global_set.object_detect_net_set.this_net.w, g_global_set.object_detect_net_set.this_net.h);

	//获取开始时间
	double this_time = get_time_point();

	//开始预测
	network_predict(g_global_set.object_detect_net_set.this_net, resize_data.data);

	//计算预测需要的时间
	detect_info.detect_time = ((double)get_time_point() - this_time) / 1000;

	//获取方框数量
	int box_number;
	detection* detection_data = get_network_boxes(&g_global_set.object_detect_net_set.this_net,
		original_data.w, original_data.h, detect_info.thresh, detect_info.hier_thresh, 0, 1, &box_number, 0);

	//非极大值抑制
	do_nms_sort(detection_data, box_number, g_global_set.object_detect_net_set.classes, detect_info.nms);

	//获取有效的方框数量
	int useble_box = 0;
	detection_with_class* detector_data = get_actual_detections(detection_data,
		box_number, detect_info.thresh, &useble_box, g_global_set.object_detect_net_set.classes_name);

	//对每一个对象
	for (int i = 0; i < useble_box; i++)
	{
		//获取类型
		int index = detector_data[i].best_class;

		//获取置信度
		int confid = detector_data[i].det.prob[detector_data[i].best_class] * 100;

		//信息
		char format[1024];
		sprintf(format, "%s %d", g_global_set.object_detect_net_set.classes_name[index], confid);

		//如果是车牌 且 车牌识别网络有加载
		if (is_object_car_id(index) && g_global_set.car_id_identify_net.initizlie)
		{
			//分析车牌
			int car_id[7];
			detect_info.identify_time = analyse_car_id(rgb_data, detector_data[i].det.bbox, car_id);

			char temp[default_char_size];
			sprintf(temp, "  %d%d-%d%d%d%d%d", car_id[0], car_id[1], car_id[2], car_id[3], car_id[4], car_id[5], car_id[6]);
			strcat(format, temp);
		}

		//绘制
		draw_boxs_and_classes(opencv_data, detector_data[i].det.bbox, format);
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

double analyse_car_id(cv::Mat& picture_data, box box_info, int* car_id_info)
{
	//获取车牌实际位置
	calc_trust_box(box_info, picture_data.cols, picture_data.rows);

	//把车牌区域拿出来
	cv::Mat roi = picture_data({ (int)box_info.x,(int)box_info.y,(int)box_info.w - (int)box_info.x,(int)box_info.h - (int)box_info.y });

	//验证车牌位置
	check_car_id_rect(roi);

	//总时间
	double all_time = 0.0;

	//类别数量
	int outputs = g_global_set.car_id_identify_net.this_net.outputs;

	//大小 
	int width = g_global_set.car_id_identify_net.this_net.w;
	int height = g_global_set.car_id_identify_net.this_net.h;

	//车牌有7位数
	for (int i = 0; i < 7; i++)
	{
		//获取一位的数据
		cv::Mat data = get_car_id_data_from_index(roi, i);

		//转化
		image temp;
		mat_translate_image(data, temp);
		image resized = resize_min(temp, width);
		image r = crop_image(resized, (resized.w - width) / 2, (resized.h - height) / 2, width, height);
		
		//获取开始时间
		double this_time = get_time_point();

		//预测
		float *predictions = network_predict(g_global_set.car_id_identify_net.this_net, r.data);

		//加入总时间
		all_time += ((double)get_time_point() - this_time) / 1000.0f;

		//获取概率最高的
		get_max_car_id(predictions, outputs, car_id_info[i]);

		//释放内存
		if (r.data != temp.data) free_image(r);
		free_image(temp);
		free_image(resized);
	}

	//返回耗时
	return all_time;
}

void check_car_id_rect(cv::Mat roi)
{

}

cv::Mat get_car_id_data_from_index(cv::Mat& data, int index)
{
	//获取图像的大小
	int width = data.cols;
	int height = data.rows;

	if (index <= 1)//车牌前面的两个字符
	{
		width = width / 9 * 3;
		int per = width / 2;
		return data({ index * per,0,per,height });
	}
	else//车牌后面的五个字符
	{
		index -= 2;
		int start = width / 9 * 3;
		int length = width - start;
		int per = length / 5;
		return data({ start + (index * per),0,per,height });
	}
}

void get_max_car_id(float* predictions, int count, int& index, float* confid)
{
	//查找最大值索引
	index = 0;
	for (int i = 1; i < count; i++)
		if (predictions[index] < predictions[i]) index = i;

	//置信度
	if (confid) *confid = predictions[index];
}

void draw_boxs_and_classes(cv::Mat& picture_data, box box_info, const char* name)
{
	//计算方框位置
	int left = (box_info.x - box_info.w / 2.)*picture_data.cols;
	int right = (box_info.x + box_info.w / 2.)*picture_data.cols;
	int top = (box_info.y - box_info.h / 2.)*picture_data.rows;
	int bot = (box_info.y + box_info.h / 2.)*picture_data.rows;

	if (left < 0) left = 0;
	if (right > picture_data.cols - 1) right = picture_data.cols - 1;
	if (top < 0) top = 0;
	if (bot > picture_data.rows - 1) bot = picture_data.rows - 1;

	//计算字体大小
	float font_size = picture_data.rows / 1200.0f;
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
	cv::Point font_left{ left,top - text_size.height * 2 };
	cv::Point font_right{ right,top };
	if (font_left.y < 0) font_left.y = 1;

	//绘制字体方框
	cv::rectangle(picture_data, font_left, font_right, cv::Scalar(font_rgb[2], font_rgb[1], font_rgb[0]), thickness, 8, 0);
	cv::rectangle(picture_data, font_left, font_right, cv::Scalar(font_rgb[2], font_rgb[1], font_rgb[0]), cv::FILLED, 8, 0);

	//绘制字体
	cv::Point pos{ font_left.x, font_left.y + text_size.height };
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
	g_global_set.picture_set.make(width, height, channel + 1);

	//图片格式转化
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			g_global_set.picture_set.data[(i * width + j) * (channel + 1) + 0] = opencv_data.data[(i * width + j) * channel + 0];
			g_global_set.picture_set.data[(i * width + j) * (channel + 1) + 1] = opencv_data.data[(i * width + j) * channel + 1];
			g_global_set.picture_set.data[(i * width + j) * (channel + 1) + 2] = opencv_data.data[(i * width + j) * channel + 2];
			g_global_set.picture_set.data[(i * width + j) * (channel + 1) + 3] = 0xff;
		}
	}
}

void read_video_frame(const char* target)
{
	cv::VideoCapture cap(target);
	if (!cap.isOpened())
	{
		show_window_tip("视频文件打开失败");
		return;
	}

	cv::Mat frame;
	if (cap.read(frame)) update_picture_texture(frame);
	else show_window_tip("视频帧读取失败");

	cap.release();
}

unsigned __stdcall  analyse_video(void* prt)
{
	//转化
	video_control* control_ptr = (video_control*)prt;
	control_ptr->leave = false;

	//打开视频文件
	video_handle_info video_info;
	if (control_ptr->use_camera) video_info.cap.open(control_ptr->camera_index);
	else video_info.cap.open(control_ptr->video_path);
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
	video_info.scene_delay = &control_ptr->scene_delay;

	//视频宽度 高度
	video_info.video_width = control_ptr->video_size[0];
	video_info.video_height = control_ptr->video_size[1];

	//创建线程
	int detect_threas = control_ptr->detect_count;
	HANDLE read_handle, scene_handle;
	HANDLE* detect_handle = new HANDLE[detect_threas];
	if (!detect_handle)
	{
		show_window_tip("申请线程句柄内存失败");
		video_info.clear();
		return 0;
	}

	//创建读取线程
	read_handle = (HANDLE)_beginthreadex(NULL, 0, read_frame_proc, &video_info, 0, NULL);
	check_serious_error(read_handle, "创建读取视频帧线程失败");

	//创建场景线程
	scene_handle = (HANDLE)_beginthreadex(NULL, 0, scene_event_proc, &video_info, 0, NULL);
	check_serious_error(scene_handle, "创建场景检测线程失败");

	//创建检测线程
	for (int i = 0; i < detect_threas; i++)
	{
		detect_handle[i] = (HANDLE)_beginthreadex(NULL, 0, prediction_frame_proc, &video_info, 0, NULL);
		check_serious_error(detect_handle[i], "创建检测视频帧线程失败");
	}

	//计算fps
	double fps = 0.0, before = 0.0;
	char fps_char[default_char_size];

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
			//计算fps
			double after = get_time_point();
			double curr = 1000000. / (after - before);
			fps = fps * 0.9 + curr * 0.1;
			before = after;
			sprintf(fps_char, "fps is : %d", (int)fps);
			cv::putText(video_ptr->original_frame, fps_char, cv::Point(10, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 255), 1, 8);

			//绘制区域
			for (auto it : g_global_set.mask_list)
			{
				//获取方框
				box this_box = it.get_box();

				//转化为真实位置
				calc_trust_box(this_box, video_ptr->original_frame.cols, video_ptr->original_frame.rows);

				//绘制人物方框
				cv::rectangle(video_ptr->original_frame, cv::Point(this_box.x, this_box.y), cv::Point(this_box.w, this_box.h), cv::Scalar(it.rect_color.w * 255, it.rect_color.y * 255, it.rect_color.x * 255), 1, 8, 0);
			}

			//显示结果
			cv::imshow(control_ptr->video_path, video_ptr->original_frame);
		}

		//显示
		cv::waitKey(*video_info.show_delay);

		//释放视频帧内存
		if (video_ptr)
		{
			//视频帧数据
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
	WaitForSingleObject(scene_handle, 1000 * 3);
	WaitForMultipleObjects(detect_threas, detect_handle, TRUE, 1000 * 3);

	//释放相关内存
	video_info.clear();

	//释放线程句柄内存
	delete[] detect_handle;

	//设置退出标志
	control_ptr->leave = true;
	return 0;
}

unsigned __stdcall read_frame_proc(void* prt)
{
	//获取
	video_handle_info* video_info = (video_handle_info*)prt;

	//获取视频宽度 高度
	int width = video_info->video_width;
	int height = video_info->video_height;

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

			//如果我们设置视频大小
			if (width && height) cv::resize(video_ptr->original_frame, video_ptr->original_frame, cv::Size(width, height));

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
	int input_width = g_global_set.object_detect_net_set.this_net.w;
	int input_height = g_global_set.object_detect_net_set.this_net.h;
	int input_channel = g_global_set.object_detect_net_set.this_net.c;

	//类型数量
	int classes = g_global_set.object_detect_net_set.classes;

	//类型名称
	char** names = g_global_set.object_detect_net_set.classes_name;

	//获取阈值信息
	const float& thresh = g_global_set.video_detect_set.thresh;
	const float& hier_thresh = g_global_set.video_detect_set.hier_thresh;
	const float& nms = g_global_set.video_detect_set.nms;

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
			network_predict(g_global_set.object_detect_net_set.this_net, original_data.data);

			//获取方框数量
			int box_count = 0;
			detection* detect_data = get_network_boxes(&g_global_set.object_detect_net_set.this_net, input_width, input_height, thresh, hier_thresh, 0, 1, &box_count, 0);

			//进行非极大值抑制
			do_nms_sort(detect_data, box_count, classes, nms);

			//操作每一个方框
			for (int i = 0; i < box_count; i++)
			{
				//临时字符串
				char temp[default_char_size];

				//每一个类检测
				for (int j = 0; j < classes; j++)
				{
					if (detect_data[i].prob[j] > thresh)
					{
						//绘制
						sprintf(temp, "%s %d", names[j], (int)(detect_data[i].prob[j] * 100.0f));
						draw_boxs_and_classes(video_ptr->original_frame, detect_data[i].bbox, temp);
					}
				}
			}

			//放入列表让场景检测
			video_info->scene_datas.push_back({ detect_data, box_count ,video_ptr->original_frame.cols,video_ptr->original_frame.rows});

			//能进行显示通知
			video_ptr->display = true;

			//释放图像数据
			free_image(original_data);
			picture_data.release();
			rgb_data.release();
		}

		Sleep(*video_info->detect_delay);//暂停
	}

	//设置检测线程状态
	video_info->detect_frame = false;
	return 0;
}

unsigned __stdcall scene_event_proc(void* prt)
{
	//转化
	video_handle_info* video_info = (video_handle_info*)prt;

	//场景设置相关
	bool &human_traffic = g_global_set.secne_set.human_count.enable;
	bool &car_traffic = g_global_set.secne_set.car_count.enable;
	bool &occupy_bus_lane = g_global_set.secne_set.bus_datas.enable;

	//阈值
	float &thresh = g_global_set.video_detect_set.thresh;

	//获取类数量
	int classes = g_global_set.object_detect_net_set.classes;

	//大小
	int width, height;

	//检测线程没退出才能工作，退出了的话我们也要跟着退出
	while (video_info->detect_frame) 
	{
		//有数据
		if (video_info->scene_datas.size())
		{
			//拿取一个
			detect_result detect_data = std::move(video_info->scene_datas[0]);
			video_info->scene_datas.erase(video_info->scene_datas.begin());

			//获取视频的大小
			width = detect_data.width;
			height = detect_data.height;

			//保存数量
			int human_count = 0;
			int car_count = 0;

			//位置保存列表
			std::vector<box> human;
			std::vector<box> car;

			//每一个方框
			for (int i = 0; i < detect_data.count; i++)
			{
				//判断是哪一个类型
				for (int j = 0; j < classes; j++)
				{
					//大于阈值
					if (detect_data.data[i].prob[j] > thresh)
					{
						//获取位置
						box b = detect_data.data[i].bbox;
						switch (j)
						{
						case object_type_car_id:
							break;
						case object_type_car:
							car_count++;
							if (occupy_bus_lane) car.push_back(b);
							break;
						case object_type_person:
							human_count++;
							break;
						case object_type_motorbike:
							break;
						case object_type_bicycle:
							break;
						case object_type_trafficlight:
							break;
						case object_type_dog:
							break;
						case object_type_bus:
							break;
						}

						//结束循环
						break;
					}
				}
			}

			//统计人流量
			if (human_traffic) calc_human_traffic(human_count);

			//统计车流量
			if (car_traffic) calc_car_traffic(car_count);

			//占用公交车道
			if (occupy_bus_lane) check_occupy_bus_lane(car, width, height);
		}

		Sleep(*video_info->detect_delay);//暂停
	}

	return 0;
}

#define CHECK_TICK(last_tick,value) if(++last_tick < value) return; else last_tick = 0;

void calc_human_traffic(int value)
{
	//2秒检测一次
	static int last_tick = 0;
	CHECK_TICK(last_tick, 30);

	//设置当前人流量
	g_global_set.secne_set.human_count.set_current_count(value);

	//上一次的人数
	static int last_count = 0;

	//递增人数
	g_global_set.secne_set.human_count.add_count(value - last_count);

	//保存人数
	last_count = value;
}

void calc_car_traffic(int value)
{
	//2秒检测一次
	static int last_tick = 0;
	CHECK_TICK(last_tick, 60);

	//设置当前车流量
	g_global_set.secne_set.car_count.set_current_count(value);

	//上一次的车流量
	static int last_count = 0;

	//递增车流量
	g_global_set.secne_set.car_count.add_count(value - last_count);

	//保存当前车流量
	last_count = value;
}

void calc_trust_box(box& b, int width, int height)
{
	int left = (b.x - b.w / 2.)*width;
	int right = (b.x + b.w / 2.)*width;
	int top = (b.y - b.h / 2.)*height;
	int bot = (b.y + b.h / 2.)*height;

	if (left < 0) left = 0;
	if (right > width - 1) right = width - 1;
	if (top < 0) top = 0;
	if (bot > height - 1) bot = height - 1;

	b.x = left;//开始位置的x
	b.y = top;//开始位置的y
	b.w = right;//结束位置的x
	b.h = bot;//结束位置的y
}

bool calc_intersect(box b1, box b2, float ratio)
{
	//交换
	auto self_swap = [](box& _b1, box& _b2)
	{
		box temp = _b1;
		_b1 = _b2;
		_b2 = temp;
	};

	//左右相交判断
	if (b1.x > b2.x) self_swap(b1, b2);
	if (b1.w <= b2.x) return false;
	float radio_w = (b1.w - b2.x) / (b2.w - b1.x);

	//上下相交判断
	if (b1.y > b2.y) self_swap(b1, b2);
	if (b1.h <= b2.y) return false;
	float radio_h = (b1.h - b2.y) / (b2.h - b1.y);

	//有一个相交率大于就判定为同一个了
	if (radio_w > ratio || radio_h > ratio) return true;
	else return false;
}

bool calc_same_rect(std::vector<box>& b_list, box& b)
{
	for (int i = 0; i < b_list.size(); i++) if (calc_intersect(b_list[i], b)) return true;
	return false;
}

void check_occupy_bus_lane(std::vector<box> b, int width, int height)
{
	//2秒检测一次
	static int last_tick = 0; 
	CHECK_TICK(last_tick, 30);

	//与公交车道相交的车辆
	std::vector<box> regions;

	//遍历区域
	for (auto& it : g_global_set.mask_list)
	{
		//非公交车道
		if(it.type != region_bus_lane) continue;

		//公交车道区域
		box bus_region = it.get_box();
		calc_trust_box(bus_region, width, height);

		//车辆遍历
		for (auto& ls : b) calc_trust_box(ls, width, height);
		for (auto& ls : b)	if (calc_intersect(bus_region, ls)) regions.push_back(ls);
	}

	//上一帧进入的车辆
	static std::vector<box> last_regions;

	for (const auto& it : regions)
	{
		//判断这一辆车上一帧是不是被检测过了
		bool state = false;
		for (int i = 0; i < last_regions.size(); i++)
		{
			if (calc_intersect(it, last_regions[i]))//相交大于0.5就是被检测过了
			{
				last_regions.erase(last_regions.begin() + i);
				state = true;
				break;
			}
		}
		if (!state)//没有被检测过
		{
			scene_info::occupy_bus_info::bus_data temp;

			//车牌识别....

			g_global_set.secne_set.bus_datas.push_bus_data(temp);
		}
	}

	//保存
	last_regions = std::move(regions);

	/*

	//上一次车辆位置
	static std::vector<box> last_pos;

	//全部转化为真实位置
	for (auto& it : b) calc_trust_box(it, width, height);

	//遍历每一个区域
	for (auto& it : g_global_set.mask_list)
	{
		//如果是公交车道区域
		if (it.type == region_bus_lane)
		{
			//公交车道真实位置
			box region_box = it.get_box();
			calc_trust_box(region_box, width, height);

			//车辆与公交车道是否相交
			for (int i = 0; i < b.size(); i++)
			{
				//与公交车道相交
				if (calc_intersect(region_box, b[i]))
				{
					//不是同一辆
					if (!calc_same_rect(last_pos, b[i]))
					{
						car_info info;
						//车牌检测
						//........

						//获取当前时间
						time_t timep;
						time(&timep);
						struct tm *prt = gmtime(&timep);
						info.times[0] = prt->tm_year + 1900;//年
						info.times[1] = prt->tm_mon + 1;//月
						info.times[2] = prt->tm_mday;//日
						info.times[3] = prt->tm_hour + 8;//时
						info.times[4] = prt->tm_min;//分
						info.times[5] = prt->tm_sec;//秒
						g_global_set.secne_set.occupy_bus_list.push_back(std::move(info));
					}
				}
				else//不相交 删除之
				{
					b.erase(b.begin() + i);
					i--;
				}
			}
		}
	}
	last_pos = std::move(b);
	*/
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

	float thresh = 0.5f;
	float hier_thresh = 0.5f;
	float nms = 0.5f;

	//遍历每一张图片
	for (int i = 0; i < picture_list.size(); i++)
	{
		if (i && i % 500 == 0) printf("图片数量 %d    完成数量 : %d \n", picture_list.size(), i);

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
		detection *dets = get_network_boxes(&net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, 0);

		//非极大值抑制
		do_nms_sort(dets, nboxes, net.layers[net.n - 1].classes, nms);

		//获取有效方框
		int detections_num;
		detection_with_class* selected_detections = get_actual_detections(dets, nboxes, thresh, &detections_num, names);

		//有对象就创建文件
		if (detections_num)
		{
			//构建文件名字
			std::string file_name = jpg_name.substr(0, jpg_name.rfind('.'));
			file_name += ".txt";

			int useful_count = 0;

			//写入位置
			std::fstream file(file_name, std::fstream::out | std::fstream::trunc);//注意修改这里
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
							sprintf(format, "%d %f %f %f %f\n", it.second, b.x, b.y, b.w, b.h);//注意修改这里
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

			//没有对象就删除txt文件
			if (useful_count == 0) DeleteFileA(file_name.c_str());//记住修改这里
		}

		//释放内存
		free_detections(dets, nboxes);
		free_image(im);
		free_image(sized);
		free(selected_detections);
	}

	//释放类名
	free_ptrs((void**)names, names_size);

	//释放网络
	free_network(net);
}
