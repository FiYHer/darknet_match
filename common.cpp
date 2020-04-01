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

	//������ʾ����
	int gpu_index;
	cudaGetDeviceCount(&gpu_index);
	check_serious_error(gpu_index, "����Կ�ʧ��");
	cuda_set_device(gpu_index - 1);
	CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));

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
		original_data.w, original_data.h, .25f, .5f, 0, 1, &g_global_set.box_number, 0);

	//�Ǽ���ֵ����
	do_nms_sort(g_global_set.detection_data, g_global_set.box_number, g_global_set.net_set.classes, .4f);

	//��ȡ��Ч�ķ�������
	int useble_box = 0;
	detection_with_class* detector_data = get_actual_detections(g_global_set.detection_data,
		g_global_set.box_number, .25f, &useble_box, g_global_set.net_set.classes_name);

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
	video_info.cap.open(0);
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

	//�ӳ�
	int& delay = g_global_set.show_video_delay;

	double fps = 0, before = 0;

	//ѭ����ʾ��Ƶ֡
	while (true)
	{
		video_frame_info* video_ptr = nullptr;

		//��ȡ����Ȩ
		video_info.entry();

		//�������Ƶ֡
		if (video_info.detect_datas.size())
		{
			//������ȡ��һ����Ƶ֡
			video_ptr = video_info.detect_datas[0];
			
			//����ʾ����ʾ
			if (video_ptr->display) video_info.detect_datas.erase(video_info.detect_datas.begin());
			else video_ptr = nullptr;
		}

		//�ͷſ���Ȩ
		video_info.leave();

		//����н����
		if (video_ptr)
		{
			double after = get_time_point();    // more accurate time measurements
			double curr = 1000000. / (after - before);
			fps = fps * 0.9 + curr * 0.1;
			before = after;
			static char fps_char[default_char_size];
			sprintf(fps_char, "Fps is : %.2lf", fps);
			cv::putText(video_ptr->original_frame, fps_char, cv::Point(50, 50), cv::FONT_HERSHEY_COMPLEX, 2, cv::Scalar(255, 0, 0));

			//��ʾ���
			cv::imshow(video_path, video_ptr->original_frame);
			cv::waitKey(delay);

			//�ͷ���Ƶ֡
			video_ptr->original_frame.release();
		}
		else if(!video_info.detect_frame) break;
		else Sleep(delay);
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

	//�ӳ٣�
	int& delay = g_global_set.read_video_delay;

	//��ʼѭ����ȡ��Ƶ֡
	while (video_info->read_frame)
	{
		//Ҫ���˳�
		if(video_info->break_state) break;

		//��ֹ��ȡ̫���ⲻ������ռ��̫���ڴ�
		if (video_info->max_frame_count > video_info->detect_datas.size())
		{
			//�����ڴ�
			video_frame_info* video_ptr = new video_frame_info;
			check_serious_error(video_ptr, "���뱣����Ƶ֡�ڴ�ʧ��");
			video_ptr->detecting = false;//û���
			video_ptr->display = false;//������ʾ

			//��ȡһ֡��Ƶ
			if (!video_info->cap.read(video_ptr->original_frame)) break;

			//����ȡ�ÿ���Ȩ
			video_info->entry();

			//��ֵָ��
			video_info->detect_datas.push_back(video_ptr);

			//�ͷſ���Ȩ
			video_info->leave();
		}

		//�Ź�CPU
		Sleep(delay);
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

	//�ӳ�
	int& delay = g_global_set.detect_video_delay;

	//��ʼ�����Ƶ֡
	while (video_info->detect_frame)
	{
		//Ҫ���˳�
		if(video_info->break_state) break;

		video_frame_info* video_ptr = nullptr;

		//��ȡ����Ȩ
		video_info->entry();

		//�ҵ�һ����û��ʼ������Ƶ֡
		for (int i = 0; i < video_info->detect_datas.size(); i++)
		{
			if (video_info->detect_datas[i]->detecting == false)
			{
				video_info->detect_datas[i]->detecting = true;
				video_ptr = video_info->detect_datas[i];
				break;
			}
		}

		//�ͷſ���Ȩ
		video_info->leave();

		//����Ƶ֡�����ǹ���
		if (video_ptr)
		{
			//��������
			cv::Mat picture_data = cv::Mat(input_height, input_width, CV_8UC(input_channel));
			cv::resize(video_ptr->original_frame, picture_data, picture_data.size(), 0, 0, cv::INTER_LINEAR);

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

			//����ÿһ������
			for (int i = 0; i < useful_box; i++)
			{
				object_info object;

				//��ȡ�������
				object.class_index = class_data[i].best_class;

				//��ȡ���Ŷ�
				object.confidence = class_data[i].det.prob[object.class_index] * 100.0f;

				//��ȡ����λ��
				get_object_rect(video_ptr->original_frame.cols, video_ptr->original_frame.rows, class_data[i].det.bbox, object);

				//���Ʒ���
				draw_object_rect(video_ptr->original_frame, object.left, object.top, object.right, object.down);
			}

			//�ܽ�����ʾ֪ͨ
			video_ptr->display = true;

			//�ͷ�ͼ������
			free_image(original_data);
			picture_data.release();

			//�ͷ��ڴ�
			free_detections(detection_data, box_count);
			free(class_data);

			//�ӳ�
			Sleep(delay);
		}
		else if(!video_info->read_frame) break;//��Ƶ��ȡ����
		else Sleep(delay);//û��ͼ�����ͣ1����
	}
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

//��ͼƬת��Ϊ��ǩ
void picture_to_label(const char* path, std::vector<std::string>& class_name, int index)
{
	//��ǩ�ļ�����
	int names_size = 0;

	//��ȡ��ǩ����
	char **names = get_labels_custom("h:\\delete\\coco.names", &names_size); //get_labels(name_list);

	//��ȡ��������
	network net = parse_network_cfg_custom("h:\\delete\\yolov3.cfg", 1, 1); // set batch=1

	//����Ȩ���ļ�
	load_weights(&net, "h:\\delete\\yolov3.weights");

	//�ںϾ��
	fuse_conv_batchnorm(net);

	//���������Ȩ��
	calculate_binary_weights(net);

	//��ȡͼƬ�ļ�
	std::vector<std::string> picture_list = get_path_from_str(path, "*.jpg");

	printf("ͼƬ������ %d \n", picture_list.size());

	//����ÿһ��ͼƬ
	for (int i = 0; i < picture_list.size(); i++)
	{
		if (i && i % 100 == 0) printf("������� : %d \n", i);

		//��������
		std::string jpg_name = path;
		jpg_name += "\\" + picture_list[i];

		//����ͼ���ת����С
		image im = load_image((char*)jpg_name.c_str(), 0, 0, net.c);
		image sized = resize_image(im, net.w, net.h);

		//����Ԥ��
		network_predict(net, sized.data);

		//��ȡ������Ϣ
		int nboxes = 0;
		detection *dets = get_network_boxes(&net, im.w, im.h, .25f, .5f, 0, 1, &nboxes, 0);

		//�Ǽ���ֵ����
		do_nms_sort(dets, nboxes, net.layers[net.n - 1].classes, .4f);

		//��ȡ��Ч����
		int detections_num;
		detection_with_class* selected_detections = get_actual_detections(dets, nboxes, .25f, &detections_num, names);

		//��ÿһ������
		for (int j = 0; j < detections_num; j++)
		{
			//��ȡ������
			int class_index = selected_detections[j].best_class;

			//��������һ��
			auto result = std::find(class_name.begin(), class_name.end(), names[class_index]);
			if (result != class_name.end())
			{
				//��ȡλ��
				box b = selected_detections[j].det.bbox;

				//�����ļ�����
				std::string file_name = jpg_name.substr(0, jpg_name.rfind('.'));
				file_name += ".txt";

				//д��λ��
				std::fstream file(file_name, std::fstream::out);
				if (file.is_open())
				{
					char format[1024];
					sprintf(format, "%d %f %f %f %f", index, b.x, b.y, b.w, b.h);
					file.write(format, strlen(format));
					file.close();
				}
			}
		}

		//�ͷ��ڴ�
		free_detections(dets, nboxes);
		free_image(im);
		free_image(sized);
		free(selected_detections);
	}
}
