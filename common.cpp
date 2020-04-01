#include "common.h"

void check_serious_error(bool state, const char* show_str /*= ""*/)
{
	if (!state)
	{
		char buffer[1024];
		sprintf(buffer, "�������ش���!\t ������ʾ:%s\t �ļ�:%s\t ����:%d\t \n", show_str, __FILE__, __LINE__);
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
		show_window_tip("�ļ���ȡʧ��");
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
		show_window_tip("���粻��Ҫ�ٴγ�ʼ��");
		return true;
	}

	//��ȡ��ǩ����
	int classes_number;
	g_global_set.net_set.classes_name = get_labels_custom(g_global_set.net_set.names_path, &classes_number);
	check_serious_error(classes_number == g_global_set.net_set.classes, "��ǩ��������Ӧ");

	//��ȡ��������
	g_global_set.net_set.match_net = parse_network_cfg_custom(g_global_set.net_set.cfg_path, 1, 1);

	//����Ȩ���ļ�
	load_weights(&g_global_set.net_set.match_net, g_global_set.net_set.weights_path);

	//�ںϾ��
	fuse_conv_batchnorm(g_global_set.net_set.match_net);

	//���������Ȩ��
	calculate_binary_weights(g_global_set.net_set.match_net);

	//�ٴμ���ǩ
	check_serious_error(g_global_set.net_set.match_net.layers[g_global_set.net_set.match_net.n - 1].classes == classes_number, "��yolo���ǩ������");

	//����״̬
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
	//����ͼƬ����
	opencv_data = cv::imread(target);
	if (opencv_data.empty())
	{
		show_window_tip("ͼƬ�ļ���ȡʧ��");
		return;
	}

	//��ȡͼƬ��Ϣ
	int width = opencv_data.cols, height = opencv_data.rows, channel = opencv_data.channels();

	//ת��Ϊimage�ṹ˳���ֵ��
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
	//����ͼƬ
	image original_data;
	cv::Mat opencv_data;
	read_picture_data(target, original_data, opencv_data);

	//��ͼƬ���ݽ�������
	image resize_data = resize_image(original_data, g_global_set.net_set.match_net.w, g_global_set.net_set.match_net.h);

	//��ȡ��ʼʱ��
	double this_time = get_time_point();

	//��ʼԤ��
	network_predict(g_global_set.net_set.match_net, resize_data.data);

	//����Ԥ����Ҫ��ʱ��
	g_global_set.detection_time = ((double)get_time_point() - this_time) / 1000;

	//�ͷ���һ�μ��õ����ڴ�
	if (g_global_set.detection_data)
	{
		free_detections(g_global_set.detection_data, g_global_set.box_number);
		g_global_set.detection_data = nullptr;
		g_global_set.box_number = 0;
	}

	//��ȡ��������
	g_global_set.detection_data = get_network_boxes(&g_global_set.net_set.match_net,
		original_data.w, original_data.h, .45f, .5f, 0, 1, &g_global_set.box_number, 0);

	//�Ǽ���ֵ����
	do_nms_sort(g_global_set.detection_data, g_global_set.box_number, g_global_set.net_set.classes, .4f);

	//��ȡ��Ч�ķ�������
	int useble_box = 0;
	detection_with_class* detector_data = get_actual_detections(g_global_set.detection_data,
		g_global_set.box_number, .45f, &useble_box, g_global_set.net_set.classes_name);

	//�����һ�εĶ���λ������
	object.clear();

	//��ÿһ������
	for (int i = 0; i < useble_box; i++)
	{
		object_info temp_object;
		temp_object.class_index = detector_data[i].best_class;//��ȡ����
		temp_object.confidence = detector_data[i].det.prob[detector_data[i].best_class] * 100;//��ȡ���Ŷ�

		//����λ����Ϣ
		get_object_rect(original_data.w, original_data.h, detector_data[i].det.bbox, temp_object);

		//�����б�
		object.push_back(std::move(temp_object));

		//���Ʒ���
		draw_object_rect(opencv_data, temp_object.left, temp_object.top, temp_object.right, temp_object.down);
	}

	//�ͷ��ڴ�
	free_image(original_data);
	free_image(resize_data);
	free(detector_data);

	//opencv������ʾ
	if (!show_type)
	{
		cv::imshow(target, opencv_data);
		cv::waitKey(1);
	}

	//��������������
	update_picture_texture(opencv_data);

	//�������
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
	//��ȡͼƬ��Ϣ
	int width = opencv_data.cols, height = opencv_data.rows, channel = opencv_data.channels();
	g_global_set.width = width;
	g_global_set.height = height;
	g_global_set.channel = channel + 1;

	//�����ǰ��ͼƬ����
	if (g_global_set.picture_data)
	{
		delete[] g_global_set.picture_data;
		g_global_set.picture_data = nullptr;
	}

	//�����µĿռ䱣��ͼƬ����,ΪʲôҪͨ����1����Ϊ��RGBA��ʽ
	g_global_set.picture_data = new unsigned char[width * height * (channel + 1)];
	check_serious_error(g_global_set.picture_data, "����ͼƬ�ռ�ʧ��");

	//ͼƬ��ʽת��
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
	//����Ƶ�ļ�
	video_handle_info video_info;
	video_info.cap.open(video_path);
	if (!video_info.cap.isOpened())
	{
		show_window_tip("����Ƶ�ļ�ʧ��");
		return;
	}

	//��ʼ�������Ϣ
	video_info.initialize();

	//�����߳�
	int read_threads = g_global_set.video_read_frame_threads;
	int detect_threas = g_global_set.video_detect_frame_threads;
	HANDLE* read_table = new HANDLE[read_threads];
	HANDLE* detect_table = new HANDLE[detect_threas];
	check_serious_error(read_table && detect_table, "�����߳̾���ڴ�ʧ��");
	for (int i = 0; i < read_threads; i++)
	{
		read_table[i] = (HANDLE)_beginthreadex(NULL, 0, read_frame_proc, &video_info, 0, NULL);
		check_serious_error(read_table[i], "������ȡ��Ƶ֡�߳�ʧ��");
	}
	for (int i = 0; i < detect_threas; i++)
	{
		detect_table[i] = (HANDLE)_beginthreadex(NULL, 0, prediction_frame_proc, &video_info, 0, NULL);
		check_serious_error(detect_table[i], "���������Ƶ֡�߳�ʧ��");
	}

	//ѭ����ʾ��Ƶ֡
	while (true)
	{
		//����н����
		if (video_info.useful_show_count)
		{
			//��ȡ���
			video_frame_info frame_info;
			video_info.entry();
			frame_info = std::move(video_info.decect_data.front());//��ȥ��һ��
			video_info.decect_data.erase(video_info.decect_data.begin());//�����Ƴ�
			video_info.useful_show_count--;//����ʾ��Ƶ֡����
			video_info.leave();

			//��ʾ���
			cv::imshow(video_path, frame_info.original_frame);
			cv::waitKey(1);
		}
		else if(!video_info.detect_frame) break;
		else Sleep(1);
	}

	//�رմ���
	cv::destroyWindow(video_path);

	//�ȴ��̵߳�ȫ���˳�
	WaitForMultipleObjects(read_threads, read_table, TRUE, INFINITE);
	WaitForMultipleObjects(detect_threas, detect_table, TRUE, INFINITE);

	//�ͷ�
	video_info.clear();
}

unsigned __stdcall read_frame_proc(void* prt)
{
	//��ȡ
	video_handle_info* video_info = (video_handle_info*)prt;

	//��ʼѭ����ȡ��Ƶ֡
	while (video_info->read_frame)
	{
		//Ҫ���˳�
		if(video_info->break_state) break;

		//��ȡһ֡��Ƶ
		video_frame_info video;
		if (!video_info->cap.read(video.original_frame)) break;

		//����ȡ�ÿ���Ȩ
		video_info->entry();

		//�����ƶ����캯��
		video_info->decect_data.push_back(std::move(video));

		//��ʾԤ���߳�ȥԤ�����Ƶ֡
		video_info->useful_detect_count++;

		//�ͷſ�����
		video_info->leave();

		//�Ź�CPU
		Sleep(10);
	}
	video_info->read_frame = false;
	return 0;
}

unsigned __stdcall prediction_frame_proc(void* prt)
{
	//ת��
	video_handle_info* video_info = (video_handle_info*)prt;

	//����������Ⱥ͸߶�
	int input_width = g_global_set.net_set.match_net.w;
	int input_height = g_global_set.net_set.match_net.h;
	int input_channel = g_global_set.net_set.match_net.c;

	//��������
	int classes = g_global_set.net_set.classes;

	//��ʼ�����Ƶ֡
	while (video_info->detect_frame)
	{
		//Ҫ���˳�
		if(video_info->break_state) break;

		//�ж��Ƿ���Դ��Ƶ֡����ȥԤ��
		if (video_info->useful_detect_count)
		{
			//��ȡ����Ȩ���ȡһ֡ͼ��
			video_info->entry();

			//ָ�����
			video_frame_info* video = &video_info->decect_data[video_info->useful_show_count];

			//�ɼ����Ƶ֡��������
			video_info->useful_detect_count--;
		
			//��������
			cv::Mat picture_data = cv::Mat(input_height, input_width, CV_8UC(input_channel));
			cv::resize(video->original_frame, picture_data, picture_data.size(), 0, 0, cv::INTER_LINEAR);
			//cv::cvtColor(picture_data, picture_data, cv::COLOR_RGB2BGR);

			//�ͷſ���Ȩ
			video_info->leave();

			//��ͼ��ת��
			image original_data;
			mat_translate_image(picture_data, original_data);

			//����Ԥ��
			network_predict(g_global_set.net_set.match_net, original_data.data);

			//��ȡ��������
			int box_count = 0;
			detection* detection_data = get_network_boxes(&g_global_set.net_set.match_net, input_width, input_height, .45f, .45f, 0, 1, &box_count, 0);

			//���зǼ���ֵ����
			do_nms_sort(detection_data, box_count, classes, .4f);

			//��ȡ������ֵ�ķ���
			int useful_box = 0;
			detection_with_class* class_data = get_actual_detections(detection_data, box_count, .45f, &useful_box, g_global_set.net_set.classes_name);

			//ȡ�ÿ���Ȩ
			video_info->entry();

			////����ÿһ������
			//for (int i = 0; i < useful_box; i++)
			//{
			//	//��ȡ�������
			//	video->object.class_index = class_data[i].best_class;

			//	//��ȡ���Ŷ�
			//	video->object.confidence = class_data[i].det.prob[video->object.class_index] * 100.0f;

			//	//��ȡ����λ��
			//	get_object_rect(video->original_frame.cols, video->original_frame.rows, class_data[i].det.bbox, video->object);

			//	//���Ʒ���
			//	draw_object_rect(video->original_frame,
			//		video->object.left,
			//		video->object.top,
			//		video->object.right,
			//		video->object.down);
			//}

			//Ԥ������˸���Ƶ֡����ʾ���Խ�����ʾ��
			video_info->useful_show_count++;

			//�ͷſ���Ȩ
			video_info->leave();

			//�ͷ�ͼ������
			free_image(original_data);
			picture_data.release();

			//�ͷ��ڴ�
			free_detections(detection_data, box_count);
			free(class_data);
		}
		else if(!video_info->read_frame) break;//��Ƶ��ȡ����
		else Sleep(1);//û��ͼ�����ͣ1����
	}
	video_info->detect_frame = false;
	return 0;
}

