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

bool initialize_net()
{
	if (g_global_set.net_set.initizlie)
	{
		show_window_tip("网络不需要再次初始化");
		return true;
	}

	//读取标签数据
	int classes_number;
	g_global_set.net_set.classes_name = get_labels_custom(g_global_set.net_set.names_path, &classes_number);
	check_serious_error(classes_number == g_global_set.net_set.classes, "标签数量不对应");

	//读取网络数据
	g_global_set.net_set.match_net = parse_network_cfg_custom(g_global_set.net_set.cfg_path, 1, 1);

	//加载权重文件
	load_weights(&g_global_set.net_set.match_net, g_global_set.net_set.weights_path);

	//融合卷积
	fuse_conv_batchnorm(g_global_set.net_set.match_net);

	//计算二进制权重
	calculate_binary_weights(g_global_set.net_set.match_net);

	//再次检测标签
	check_serious_error(g_global_set.net_set.match_net.layers[g_global_set.net_set.match_net.n - 1].classes == classes_number, "和yolo层标签数不符");

	//设置状态
	g_global_set.net_set.initizlie = true;
	return true;
}

void clear_net()
{
	if (g_global_set.net_set.classes_name) free_ptrs((void**)g_global_set.net_set.classes_name, g_global_set.net_set.classes);
	free_network(g_global_set.net_set.match_net);
	g_global_set.net_set.initizlie = false;
}

void read_picture_data(const char* target, image& picture_data, cv::Mat& opencv_data)
{
	//记载图片数据
	opencv_data = cv::imread(target);
	if (opencv_data.empty())
	{
		show_window_tip("图片文件读取失败");
		return;
	}

	//获取图片信息
	int width = opencv_data.cols, height = opencv_data.rows, channel = opencv_data.channels();

	//转化为image结构顺便均值化
	mat_translate_image(opencv_data, picture_data);
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

void analyse_picture(const char* target, std::vector<object_info>& object, int show_type /*= 0*/)
{
	//加载图片
	image original_data;
	cv::Mat opencv_data;
	read_picture_data(target, original_data, opencv_data);

	//将图片数据进行缩放
	image resize_data = resize_image(original_data, g_global_set.net_set.match_net.w, g_global_set.net_set.match_net.h);

	//获取开始时间
	double this_time = get_time_point();

	//开始预测
	network_predict(g_global_set.net_set.match_net, resize_data.data);

	//计算预测需要的时间
	g_global_set.detection_time = ((double)get_time_point() - this_time) / 1000;

	//释放上一次检测得到的内存
	if (g_global_set.detection_data)
	{
		free_detections(g_global_set.detection_data, g_global_set.box_number);
		g_global_set.detection_data = nullptr;
		g_global_set.box_number = 0;
	}

	//获取方框数量
	g_global_set.detection_data = get_network_boxes(&g_global_set.net_set.match_net,
		original_data.w, original_data.h, .45f, .5f, 0, 1, &g_global_set.box_number, 0);

	//非极大值抑制
	do_nms_sort(g_global_set.detection_data, g_global_set.box_number, g_global_set.net_set.classes, .4f);

	//获取有效的方框数量
	int useble_box = 0;
	detection_with_class* detector_data = get_actual_detections(g_global_set.detection_data,
		g_global_set.box_number, .45f, &useble_box, g_global_set.net_set.classes_name);

	//清空上一次的对象位置数据
	object.clear();

	//对每一个对象
	for (int i = 0; i < useble_box; i++)
	{
		object_info temp_object;
		temp_object.class_index = detector_data[i].best_class;//获取类型
		temp_object.confidence = detector_data[i].det.prob[detector_data[i].best_class] * 100;//获取置信度

		//计算位置信息
		get_object_rect(original_data.w, original_data.h, detector_data[i].det.bbox, temp_object);

		//加入列表
		object.push_back(std::move(temp_object));

		//绘制方框
		draw_object_rect(opencv_data, temp_object.left, temp_object.top, temp_object.right, temp_object.down);
	}

	//释放内存
	free_image(original_data);
	free_image(resize_data);
	free(detector_data);

	//opencv界面显示
	if (!show_type)
	{
		cv::imshow(target, opencv_data);
		cv::waitKey(1);
	}

	//更新置纹理数据
	update_picture_texture(opencv_data);

	//清空数据
	opencv_data.release();
}

void get_object_rect(int width, int height, box& box_pos, object_info& object)
{
	int left = (box_pos.x - box_pos.w / 2.)*width;
	int right = (box_pos.x + box_pos.w / 2.)*width;
	int top = (box_pos.y - box_pos.h / 2.)*height;
	int down = (box_pos.y + box_pos.h / 2.)*height;

	if (left < 0) left = 0;
	if (right > width - 1) right = width - 1;
	if (top < 0) top = 0;
	if (down > width - 1) down = width - 1;

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
	static int r = 255, g = 0, b = 0;

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

void analyse_video(const char* video_path)
{
	//打开视频文件
	video_handle_info video_info;
	video_info.cap.open(video_path);
	if (!video_info.cap.isOpened())
	{
		show_window_tip("打开视频文件失败");
		return;
	}

	//初始化相关信息
	video_info.initialize();

	//创建线程
	int read_threads = g_global_set.video_read_frame_threads;
	int detect_threas = g_global_set.video_detect_frame_threads;
	HANDLE* read_table = new HANDLE[read_threads];
	HANDLE* detect_table = new HANDLE[detect_threas];
	check_serious_error(read_table && detect_table, "申请线程句柄内存失败");
	for (int i = 0; i < read_threads; i++)
	{
		read_table[i] = (HANDLE)_beginthreadex(NULL, 0, read_frame_proc, &video_info, 0, NULL);
		check_serious_error(read_table[i], "创建读取视频帧线程失败");
	}
	for (int i = 0; i < detect_threas; i++)
	{
		detect_table[i] = (HANDLE)_beginthreadex(NULL, 0, prediction_frame_proc, &video_info, 0, NULL);
		check_serious_error(detect_table[i], "创建检测视频帧线程失败");
	}

	//循环显示视频帧
	while (true)
	{
		//如果有结果了
		if (video_info.useful_show_count)
		{
			//获取结果
			video_frame_info frame_info;
			video_info.entry();
			frame_info = std::move(video_info.decect_data.front());//拿去第一个
			video_info.decect_data.erase(video_info.decect_data.begin());//进行移除
			video_info.useful_show_count--;//可显示视频帧减少
			video_info.leave();

			//显示结果
			cv::imshow(video_path, frame_info.original_frame);
			cv::waitKey(1);
		}
		else if(!video_info.detect_frame) break;
		else Sleep(1);
	}

	//关闭窗口
	cv::destroyWindow(video_path);

	//等待线程的全部退出
	WaitForMultipleObjects(read_threads, read_table, TRUE, INFINITE);
	WaitForMultipleObjects(detect_threas, detect_table, TRUE, INFINITE);

	//释放
	video_info.clear();
}

unsigned __stdcall read_frame_proc(void* prt)
{
	//获取
	video_handle_info* video_info = (video_handle_info*)prt;

	//开始循环读取视频帧
	while (video_info->read_frame)
	{
		//要求退出
		if(video_info->break_state) break;

		//读取一帧视频
		video_frame_info video;
		if (!video_info->cap.read(video.original_frame)) break;

		//尝试取得控制权
		video_info->entry();

		//调用移动构造函数
		video_info->decect_data.push_back(std::move(video));

		//提示预测线程去预测该视频帧
		video_info->useful_detect_count++;

		//释放控制器
		video_info->leave();

		//放过CPU
		Sleep(10);
	}
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
	while (video_info->detect_frame)
	{
		//要求退出
		if(video_info->break_state) break;

		//判断是否有源视频帧可以去预测
		if (video_info->useful_detect_count)
		{
			//获取控制权后获取一帧图像
			video_info->entry();

			//指针加速
			video_frame_info* video = &video_info->decect_data[video_info->useful_show_count];

			//可检测视频帧数量减少
			video_info->useful_detect_count--;
		
			//进行缩放
			cv::Mat picture_data = cv::Mat(input_height, input_width, CV_8UC(input_channel));
			cv::resize(video->original_frame, picture_data, picture_data.size(), 0, 0, cv::INTER_LINEAR);
			//cv::cvtColor(picture_data, picture_data, cv::COLOR_RGB2BGR);

			//释放控制权
			video_info->leave();

			//将图像转换
			image original_data;
			mat_translate_image(picture_data, original_data);

			//网络预测
			network_predict(g_global_set.net_set.match_net, original_data.data);

			//获取方框数量
			int box_count = 0;
			detection* detection_data = get_network_boxes(&g_global_set.net_set.match_net, input_width, input_height, .45f, .45f, 0, 1, &box_count, 0);

			//进行非极大值抑制
			do_nms_sort(detection_data, box_count, classes, .4f);

			//获取大于阈值的方框
			int useful_box = 0;
			detection_with_class* class_data = get_actual_detections(detection_data, box_count, .45f, &useful_box, g_global_set.net_set.classes_name);

			//取得控制权
			video_info->entry();

			////操作每一个对象
			//for (int i = 0; i < useful_box; i++)
			//{
			//	//获取类别索引
			//	video->object.class_index = class_data[i].best_class;

			//	//获取置信度
			//	video->object.confidence = class_data[i].det.prob[video->object.class_index] * 100.0f;

			//	//获取方框位置
			//	get_object_rect(video->original_frame.cols, video->original_frame.rows, class_data[i].det.bbox, video->object);

			//	//绘制方框
			//	draw_object_rect(video->original_frame,
			//		video->object.left,
			//		video->object.top,
			//		video->object.right,
			//		video->object.down);
			//}

			//预测完毕了该视频帧，提示可以进行显示了
			video_info->useful_show_count++;

			//释放控制权
			video_info->leave();

			//释放图像数据
			free_image(original_data);
			picture_data.release();

			//释放内存
			free_detections(detection_data, box_count);
			free(class_data);
		}
		else if(!video_info->read_frame) break;//视频读取完了
		else Sleep(1);//没有图像就暂停1毫秒
	}
	video_info->detect_frame = false;
	return 0;
}

