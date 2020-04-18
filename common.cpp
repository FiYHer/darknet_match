#include "common.h"

void check_serious_error(bool state, const char* show_str, const char* file_pos, int line_pos)
{
	if (!state)
	{
		char buffer[default_char_size];
		sprintf(buffer, "�������ش���!\t ������ʾ:%s\t �ļ�:%s\t ����:%d\n", show_str, file_pos, line_pos);
		MessageBoxA(NULL, buffer, NULL, MB_ICONHAND | MB_OK);
		exit(-1);
	}
}

void show_window_tip(const char* str)
{
	MessageBoxA(NULL, str, "����", MB_ICONWARNING | MB_OK);
}

int get_gpu_count()
{
	//��ǰΪGPU�汾�ģ����û���Կ�֧�ֽ��޷����м���
	int gpu_count = 0;
	check_serious_error(cudaGetDeviceCount(&gpu_count) == cudaSuccess, "��ȡ�Կ�����ʧ��");
	check_serious_error(gpu_count, "û�з��ֵ����Կ�");
	return gpu_count;
}

cudaDeviceProp* get_gpu_infomation(int gpu_count)
{
	//��ȡ�Կ������Ϣ
	cudaDeviceProp* gpu_info = new cudaDeviceProp[gpu_count];
	check_serious_error(gpu_info, "�����Կ�����ڴ�ʧ��");
	for (int i = 0; i < gpu_count; i++) check_serious_error(cudaGetDeviceProperties(&gpu_info[i], i) == cudaSuccess, "��ȡ�Կ���Ϣʧ��");
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

bool initialize_object_detect_net(const char* names_file, const char* cfg_file, const char* weights_file)
{
	//��ֹ���γ�ʼ��
	if (g_global_set.object_detect_net_set.initizlie)
	{
		show_window_tip("���������粻��Ҫ�ٴγ�ʼ��");
		return true;
	}

	//�ж�names�ļ��Ƿ����
	if (!names_file || !strstr(names_file, ".names") || access(names_file, 0) == -1)
	{
		show_window_tip("names�ļ�������");
		return false;
	}

	//�ж�cfg�ļ��Ƿ����
	if (!cfg_file || !strstr(cfg_file, ".cfg") || access(cfg_file, 0) == -1)
	{
		show_window_tip("cfg�ļ�������");
		return false;
	}

	//�ж�weights�ļ��Ƿ����
	if (!weights_file || !strstr(weights_file, ".weights") || access(weights_file, 0) == -1)
	{
		show_window_tip("weights�ļ�������");
		return false;
	}

	//��ȡ��ǩ����
	int classes_number;
	g_global_set.object_detect_net_set.classes_name = get_labels_custom((char*)names_file, &classes_number);

	//��ȡ��������
	g_global_set.object_detect_net_set.this_net = parse_network_cfg_custom((char*)cfg_file, 1, 1);

	//����Ȩ���ļ�
	load_weights(&g_global_set.object_detect_net_set.this_net, (char*)weights_file);

	//batch���ںϽ������
	fuse_conv_batchnorm(g_global_set.object_detect_net_set.this_net);

	//���������Ȩ��
	calculate_binary_weights(g_global_set.object_detect_net_set.this_net);

	//����ǩ
	if (g_global_set.object_detect_net_set.this_net.layers[g_global_set.object_detect_net_set.this_net.n - 1].classes != classes_number)
	{
		show_window_tip("�����ǩ���������ļ���ǩ����һ��");
		clear_object_detect_net();
		return false;
	}

	//�����������
	g_global_set.object_detect_net_set.classes = classes_number;

	//����״̬
	g_global_set.object_detect_net_set.initizlie = true;

	//���ؽ��
	return true;
}

bool initialize_car_id_identify_net(const char* names_file, const char* cfg_file, const char* weights_file)
{
	//��ֹ���γ�ʼ��
	if (g_global_set.car_id_identify_net.initizlie)
	{
		show_window_tip("����ʶ�����粻��Ҫ�ٴγ�ʼ��");
		return true;
	}

	//�ж�names�ļ��Ƿ����
	if (!names_file || !strstr(names_file, ".names") || access(names_file, 0) == -1)
	{
		show_window_tip("names�ļ�������");
		return false;
	}

	//�ж�cfg�ļ��Ƿ����
	if (!cfg_file || !strstr(cfg_file, ".cfg") || access(cfg_file, 0) == -1)
	{
		show_window_tip("cfg�ļ�������");
		return false;
	}

	//�ж�weights�ļ��Ƿ����
	if (!weights_file || !strstr(weights_file, ".weights") || access(weights_file, 0) == -1)
	{
		show_window_tip("weights�ļ�������");
		return false;
	}

	//��������
	g_global_set.car_id_identify_net.this_net = parse_network_cfg_custom((char*)cfg_file, 1, 0);

	//����Ȩ��
	load_weights(&g_global_set.car_id_identify_net.this_net, (char*)weights_file);

	//��������Ϊ1
	set_batch_network(&g_global_set.car_id_identify_net.this_net, 1);

	//�ں�batch�������
	fuse_conv_batchnorm(g_global_set.car_id_identify_net.this_net);

	//���������Ȩ��
	calculate_binary_weights(g_global_set.car_id_identify_net.this_net);

	//��ȡ��ǩ��
	g_global_set.car_id_identify_net.classes_name = get_labels_custom((char*)names_file, &g_global_set.car_id_identify_net.classes);

	//����״̬
	g_global_set.car_id_identify_net.initizlie = true;

	return true;
}

void clear_object_detect_net()
{
	//����Ҫ�ͷ�
	if (!g_global_set.object_detect_net_set.initizlie) return;

	//�ͷű�ǩ�ַ���
	if (g_global_set.object_detect_net_set.classes_name) free_ptrs((void**)g_global_set.object_detect_net_set.classes_name, g_global_set.object_detect_net_set.classes);
	
	//�ͷ�����
	free_network(g_global_set.object_detect_net_set.this_net);

	//����״̬
	g_global_set.object_detect_net_set.initizlie = false;
}

void clear_car_id_identify_net()
{
	//����Ҫ�ͷ�
	if (!g_global_set.car_id_identify_net.initizlie) return;

	//�ͷ��ַ���
	if (g_global_set.car_id_identify_net.classes_name) free_ptrs((void**)g_global_set.car_id_identify_net.classes_name, g_global_set.car_id_identify_net.classes);

	//�ͷ�����
	free_network(g_global_set.car_id_identify_net.this_net);

	//����״̬
	g_global_set.car_id_identify_net.initizlie = false;
}

void read_picture_data(const char* target, image& picture_data, cv::Mat& opencv_data, cv::Mat& rgb_data)
{
	//����ͼƬ����
	opencv_data = cv::imread(target);
	if (opencv_data.empty()) return;
	
	//ת����ɫͨ��
	if (opencv_data.channels() == 3) cv::cvtColor(opencv_data, rgb_data, cv::COLOR_RGB2BGR);
	else if (opencv_data.channels() == 4) cv::cvtColor(opencv_data, rgb_data, cv::COLOR_RGBA2BGRA);
	else opencv_data.copyTo(rgb_data);

	//��ȡͼƬ��Ϣ
	int width = rgb_data.cols, height = rgb_data.rows, channel = rgb_data.channels();

	//ת��Ϊimage�ṹ˳���ֵ��
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
	//����ͼƬ
	image original_data;
	cv::Mat opencv_data, rgb_data;
	read_picture_data(target, original_data, opencv_data, rgb_data);
	if (opencv_data.empty())
	{
		show_window_tip("ͼƬ��ʧ��");
		return;
	}

	//��ͼƬ���ݽ�������
	image resize_data = resize_image(original_data, g_global_set.object_detect_net_set.this_net.w, g_global_set.object_detect_net_set.this_net.h);

	//��ȡ��ʼʱ��
	double this_time = get_time_point();

	//��ʼԤ��
	network_predict(g_global_set.object_detect_net_set.this_net, resize_data.data);

	//����Ԥ����Ҫ��ʱ��
	detect_info.detect_time = ((double)get_time_point() - this_time) / 1000;

	//��ȡ��������
	int box_number;
	detection* detection_data = get_network_boxes(&g_global_set.object_detect_net_set.this_net,
		original_data.w, original_data.h, detect_info.thresh, detect_info.hier_thresh, 0, 1, &box_number, 0);

	//�Ǽ���ֵ����
	do_nms_sort(detection_data, box_number, g_global_set.object_detect_net_set.classes, detect_info.nms);

	//��ȡ��Ч�ķ�������
	int useble_box = 0;
	detection_with_class* detector_data = get_actual_detections(detection_data,
		box_number, detect_info.thresh, &useble_box, g_global_set.object_detect_net_set.classes_name);

	//��ÿһ������
	for (int i = 0; i < useble_box; i++)
	{
		//��ȡ����
		int index = detector_data[i].best_class;

		//��ȡ���Ŷ�
		int confid = detector_data[i].det.prob[detector_data[i].best_class] * 100;

		//��Ϣ
		char format[1024];
		sprintf(format, "%s %d", g_global_set.object_detect_net_set.classes_name[index], confid);

		//����ǳ��� �� ����ʶ�������м���
		if (is_object_car_id(index) && g_global_set.car_id_identify_net.initizlie)
		{
			//��������
			int car_id[7];
			detect_info.identify_time = analyse_car_id(rgb_data, detector_data[i].det.bbox, car_id);

			char temp[default_char_size];
			sprintf(temp, "  %d%d-%d%d%d%d%d", car_id[0], car_id[1], car_id[2], car_id[3], car_id[4], car_id[5], car_id[6]);
			strcat(format, temp);
		}

		//����
		draw_boxs_and_classes(opencv_data, detector_data[i].det.bbox, format);
	}

	//�ͷ��ڴ�
	free_detections(detection_data, box_number);
	free_image(original_data);
	free_image(resize_data);
	free(detector_data);

	//opencv������ʾ
	if (show)
	{
		cv::imshow(target, opencv_data);
		cv::waitKey(1);
	}

	//��������������
	update_picture_texture(opencv_data);

	//�������
	opencv_data.release();
	rgb_data.release();
}

double analyse_car_id(cv::Mat& picture_data, box box_info, int* car_id_info)
{
	//��ȡ����ʵ��λ��
	calc_trust_box(box_info, picture_data.cols, picture_data.rows);

	//�ѳ��������ó���
	cv::Mat roi = picture_data({ (int)box_info.x,(int)box_info.y,(int)box_info.w - (int)box_info.x,(int)box_info.h - (int)box_info.y });

	//��֤����λ��
	check_car_id_rect(roi);

	//��ʱ��
	double all_time = 0.0;

	//�������
	int outputs = g_global_set.car_id_identify_net.this_net.outputs;

	//��С 
	int width = g_global_set.car_id_identify_net.this_net.w;
	int height = g_global_set.car_id_identify_net.this_net.h;

	//������7λ��
	for (int i = 0; i < 7; i++)
	{
		//��ȡһλ������
		cv::Mat data = get_car_id_data_from_index(roi, i);

		//ת��
		image temp;
		mat_translate_image(data, temp);
		image resized = resize_min(temp, width);
		image r = crop_image(resized, (resized.w - width) / 2, (resized.h - height) / 2, width, height);
		
		//��ȡ��ʼʱ��
		double this_time = get_time_point();

		//Ԥ��
		float *predictions = network_predict(g_global_set.car_id_identify_net.this_net, r.data);

		//������ʱ��
		all_time += ((double)get_time_point() - this_time) / 1000.0f;

		//��ȡ������ߵ�
		get_max_car_id(predictions, outputs, car_id_info[i]);

		//�ͷ��ڴ�
		if (r.data != temp.data) free_image(r);
		free_image(temp);
		free_image(resized);
	}

	//���غ�ʱ
	return all_time;
}

void check_car_id_rect(cv::Mat roi)
{

}

cv::Mat get_car_id_data_from_index(cv::Mat& data, int index)
{
	//��ȡͼ��Ĵ�С
	int width = data.cols;
	int height = data.rows;

	if (index <= 1)//����ǰ��������ַ�
	{
		width = width / 9 * 3;
		int per = width / 2;
		return data({ index * per,0,per,height });
	}
	else//���ƺ��������ַ�
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
	//�������ֵ����
	index = 0;
	for (int i = 1; i < count; i++)
		if (predictions[index] < predictions[i]) index = i;

	//���Ŷ�
	if (confid) *confid = predictions[index];
}

void draw_boxs_and_classes(cv::Mat& picture_data, box box_info, const char* name)
{
	//���㷽��λ��
	int left = (box_info.x - box_info.w / 2.)*picture_data.cols;
	int right = (box_info.x + box_info.w / 2.)*picture_data.cols;
	int top = (box_info.y - box_info.h / 2.)*picture_data.rows;
	int bot = (box_info.y + box_info.h / 2.)*picture_data.rows;

	if (left < 0) left = 0;
	if (right > picture_data.cols - 1) right = picture_data.cols - 1;
	if (top < 0) top = 0;
	if (bot > picture_data.rows - 1) bot = picture_data.rows - 1;

	//���������С
	float font_size = picture_data.rows / 1200.0f;
	cv::Size text_size = cv::getTextSize(name, cv::FONT_HERSHEY_COMPLEX_SMALL, font_size, 1, 0);

	//������ɫ
	float box_rgb[3]{ g_global_set.color_set.box_rgb[0] * 255.0f, g_global_set.color_set.box_rgb[1] * 255.0f,g_global_set.color_set.box_rgb[2] * 255.0f };
	
	//������ɫ
	float font_rgb[3]{ g_global_set.color_set.font_rgb[0] * 255.0f,g_global_set.color_set.font_rgb[1] * 255.0f ,g_global_set.color_set.font_rgb[2] * 255.0f };

	//������ϸ
	float thickness = g_global_set.color_set.thickness;

	//�������﷽��
	cv::rectangle(picture_data, cv::Point(left, top), cv::Point(right, bot), cv::Scalar(box_rgb[2], box_rgb[1], box_rgb[0]), thickness, 8, 0);

	//�������巽��λ��
	cv::Point font_left{ left,top - text_size.height * 2 };
	cv::Point font_right{ right,top };
	if (font_left.y < 0) font_left.y = 1;

	//�������巽��
	cv::rectangle(picture_data, font_left, font_right, cv::Scalar(font_rgb[2], font_rgb[1], font_rgb[0]), thickness, 8, 0);
	cv::rectangle(picture_data, font_left, font_right, cv::Scalar(font_rgb[2], font_rgb[1], font_rgb[0]), cv::FILLED, 8, 0);

	//��������
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
	//��ȡͼƬ��Ϣ
	int width = opencv_data.cols, height = opencv_data.rows, channel = opencv_data.channels();
	g_global_set.picture_set.make(width, height, channel + 1);

	//ͼƬ��ʽת��
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
		show_window_tip("��Ƶ�ļ���ʧ��");
		return;
	}

	cv::Mat frame;
	if (cap.read(frame)) update_picture_texture(frame);
	else show_window_tip("��Ƶ֡��ȡʧ��");

	cap.release();
}

unsigned __stdcall  analyse_video(void* prt)
{
	//ת��
	video_control* control_ptr = (video_control*)prt;
	control_ptr->leave = false;

	//����Ƶ�ļ�
	video_handle_info video_info;
	if (control_ptr->use_camera) video_info.cap.open(control_ptr->camera_index);
	else video_info.cap.open(control_ptr->video_path);
	if (!video_info.cap.isOpened())
	{
		show_window_tip("����Ƶ�ļ�ʧ��");
		control_ptr->leave = true;
		return 0;
	}

	//��ʼ�������Ϣ
	video_info.initialize();

	//�ӳ�����
	video_info.show_delay = &control_ptr->show_delay;
	video_info.read_delay = &control_ptr->read_delay;
	video_info.detect_delay = &control_ptr->detect_delay;
	video_info.scene_delay = &control_ptr->scene_delay;

	//��Ƶ��� �߶�
	video_info.video_width = control_ptr->video_size[0];
	video_info.video_height = control_ptr->video_size[1];

	//�����߳�
	int detect_threas = control_ptr->detect_count;
	HANDLE read_handle, scene_handle;
	HANDLE* detect_handle = new HANDLE[detect_threas];
	if (!detect_handle)
	{
		show_window_tip("�����߳̾���ڴ�ʧ��");
		video_info.clear();
		return 0;
	}

	//������ȡ�߳�
	read_handle = (HANDLE)_beginthreadex(NULL, 0, read_frame_proc, &video_info, 0, NULL);
	check_serious_error(read_handle, "������ȡ��Ƶ֡�߳�ʧ��");

	//���������߳�
	scene_handle = (HANDLE)_beginthreadex(NULL, 0, scene_event_proc, &video_info, 0, NULL);
	check_serious_error(scene_handle, "������������߳�ʧ��");

	//��������߳�
	for (int i = 0; i < detect_threas; i++)
	{
		detect_handle[i] = (HANDLE)_beginthreadex(NULL, 0, prediction_frame_proc, &video_info, 0, NULL);
		check_serious_error(detect_handle[i], "���������Ƶ֡�߳�ʧ��");
	}

	//����fps
	double fps = 0.0, before = 0.0;
	char fps_char[default_char_size];

	//ѭ����ʾ��Ƶ֡
	while (!control_ptr->leave && video_info.detect_frame)
	{
		//��Ƶָ֡��
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
			else video_ptr = nullptr;//������ʾ��ʲô������
		}

		//�ͷſ���Ȩ
		video_info.leave();

		//����н����
		if (video_ptr)
		{
			//����fps
			double after = get_time_point();
			double curr = 1000000. / (after - before);
			fps = fps * 0.9 + curr * 0.1;
			before = after;
			sprintf(fps_char, "fps is : %d", (int)fps);
			cv::putText(video_ptr->original_frame, fps_char, cv::Point(10, 30), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 255), 1, 8);

			//��������
			for (auto it : g_global_set.mask_list)
			{
				//��ȡ����
				box this_box = it.get_box();

				//ת��Ϊ��ʵλ��
				calc_trust_box(this_box, video_ptr->original_frame.cols, video_ptr->original_frame.rows);

				//�������﷽��
				cv::rectangle(video_ptr->original_frame, cv::Point(this_box.x, this_box.y), cv::Point(this_box.w, this_box.h), cv::Scalar(it.rect_color.w * 255, it.rect_color.y * 255, it.rect_color.x * 255), 1, 8, 0);
			}

			//��ʾ���
			cv::imshow(control_ptr->video_path, video_ptr->original_frame);
		}

		//��ʾ
		cv::waitKey(*video_info.show_delay);

		//�ͷ���Ƶ֡�ڴ�
		if (video_ptr)
		{
			//��Ƶ֡����
			video_ptr->original_frame.release();
			delete video_ptr;
		}
	}

	//�رմ���
	cv::destroyWindow(control_ptr->video_path);

	//�ȴ��̵߳�ȫ���˳�
	video_info.break_state = true;
	video_info.read_frame = false;
	video_info.detect_frame = false;
	WaitForSingleObject(read_handle, 1000 * 3);
	WaitForSingleObject(scene_handle, 1000 * 3);
	WaitForMultipleObjects(detect_threas, detect_handle, TRUE, 1000 * 3);

	//�ͷ�����ڴ�
	video_info.clear();

	//�ͷ��߳̾���ڴ�
	delete[] detect_handle;

	//�����˳���־
	control_ptr->leave = true;
	return 0;
}

unsigned __stdcall read_frame_proc(void* prt)
{
	//��ȡ
	video_handle_info* video_info = (video_handle_info*)prt;

	//��ȡ��Ƶ��� �߶�
	int width = video_info->video_width;
	int height = video_info->video_height;

	//��ʼѭ����ȡ��Ƶ֡
	while (video_info->read_frame && !video_info->break_state)
	{
		//��ֹ��ȡ̫���ⲻ������ռ��̫���ڴ�
		if (video_info->max_frame_count > video_info->detect_datas.size())
		{
			//�����ڴ�
			video_frame_info* video_ptr = new video_frame_info;
			check_serious_error(video_ptr, "���뱣����Ƶ֡�ڴ�ʧ��");
			video_ptr->detecting = false;//��û���
			video_ptr->display = false;//������ʾ

			//��ȡһ֡��Ƶ
			if (!video_info->cap.read(video_ptr->original_frame)) break;

			//�������������Ƶ��С
			if (width && height) cv::resize(video_ptr->original_frame, video_ptr->original_frame, cv::Size(width, height));

			//����ȡ�ÿ���Ȩ
			video_info->entry();

			//��ֵָ��
			video_info->detect_datas.push_back(video_ptr);

			//�ͷſ���Ȩ
			video_info->leave();
		}

		//�Ź�CPU
		Sleep(*video_info->read_delay);
	}

	//���ö�ȡ��־
	video_info->read_frame = false;
	return 0;
}

unsigned __stdcall prediction_frame_proc(void* prt)
{
	//ת��
	video_handle_info* video_info = (video_handle_info*)prt;

	//����������Ⱥ͸߶�
	int input_width = g_global_set.object_detect_net_set.this_net.w;
	int input_height = g_global_set.object_detect_net_set.this_net.h;
	int input_channel = g_global_set.object_detect_net_set.this_net.c;

	//��������
	int classes = g_global_set.object_detect_net_set.classes;

	//��������
	char** names = g_global_set.object_detect_net_set.classes_name;

	//��ȡ��ֵ��Ϣ
	const float& thresh = g_global_set.video_detect_set.thresh;
	const float& hier_thresh = g_global_set.video_detect_set.hier_thresh;
	const float& nms = g_global_set.video_detect_set.nms;

	//��ʼ�����Ƶ֡
	while (video_info->detect_frame && video_info->read_frame && !video_info->break_state)
	{
		//
		video_frame_info* video_ptr = nullptr;

		//��ȡ����Ȩ
		video_info->entry();

		//�ҵ�һ����û��ʼ������Ƶ֡
		for (int i = 0; i < video_info->detect_datas.size(); i++)
		{
			if (video_info->detect_datas[i]->detecting == false)//�ҵ�û������Ƶ֡
			{
				video_info->detect_datas[i]->detecting = true;//���Ϊ���ڼ��
				video_ptr = video_info->detect_datas[i];//�õ���ַ
				break;//�˳�ѭ��
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
			
			//��ɫ��ʽת��
			cv::Mat rgb_data;
			cv::cvtColor(picture_data, rgb_data, cv::COLOR_RGB2BGR);

			//��ͼ��ת��Ϊimage
			image original_data;
			mat_translate_image(rgb_data, original_data);

			//����Ԥ��
			network_predict(g_global_set.object_detect_net_set.this_net, original_data.data);

			//��ȡ��������
			int box_count = 0;
			detection* detect_data = get_network_boxes(&g_global_set.object_detect_net_set.this_net, input_width, input_height, thresh, hier_thresh, 0, 1, &box_count, 0);

			//���зǼ���ֵ����
			do_nms_sort(detect_data, box_count, classes, nms);

			//����ÿһ������
			for (int i = 0; i < box_count; i++)
			{
				//��ʱ�ַ���
				char temp[default_char_size];

				//ÿһ������
				for (int j = 0; j < classes; j++)
				{
					if (detect_data[i].prob[j] > thresh)
					{
						//����
						sprintf(temp, "%s %d", names[j], (int)(detect_data[i].prob[j] * 100.0f));
						draw_boxs_and_classes(video_ptr->original_frame, detect_data[i].bbox, temp);
					}
				}
			}

			//�����б��ó������
			video_info->scene_datas.push_back({ detect_data, box_count ,video_ptr->original_frame.cols,video_ptr->original_frame.rows});

			//�ܽ�����ʾ֪ͨ
			video_ptr->display = true;

			//�ͷ�ͼ������
			free_image(original_data);
			picture_data.release();
			rgb_data.release();
		}

		Sleep(*video_info->detect_delay);//��ͣ
	}

	//���ü���߳�״̬
	video_info->detect_frame = false;
	return 0;
}

unsigned __stdcall scene_event_proc(void* prt)
{
	//ת��
	video_handle_info* video_info = (video_handle_info*)prt;

	//�����������
	bool &human_traffic = g_global_set.secne_set.human_count.enable;
	bool &car_traffic = g_global_set.secne_set.car_count.enable;
	bool &occupy_bus_lane = g_global_set.secne_set.bus_datas.enable;

	//��ֵ
	float &thresh = g_global_set.video_detect_set.thresh;

	//��ȡ������
	int classes = g_global_set.object_detect_net_set.classes;

	//��С
	int width, height;

	//����߳�û�˳����ܹ������˳��˵Ļ�����ҲҪ�����˳�
	while (video_info->detect_frame) 
	{
		//������
		if (video_info->scene_datas.size())
		{
			//��ȡһ��
			detect_result detect_data = std::move(video_info->scene_datas[0]);
			video_info->scene_datas.erase(video_info->scene_datas.begin());

			//��ȡ��Ƶ�Ĵ�С
			width = detect_data.width;
			height = detect_data.height;

			//��������
			int human_count = 0;
			int car_count = 0;

			//λ�ñ����б�
			std::vector<box> human;
			std::vector<box> car;

			//ÿһ������
			for (int i = 0; i < detect_data.count; i++)
			{
				//�ж�����һ������
				for (int j = 0; j < classes; j++)
				{
					//������ֵ
					if (detect_data.data[i].prob[j] > thresh)
					{
						//��ȡλ��
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

						//����ѭ��
						break;
					}
				}
			}

			//ͳ��������
			if (human_traffic) calc_human_traffic(human_count);

			//ͳ�Ƴ�����
			if (car_traffic) calc_car_traffic(car_count);

			//ռ�ù�������
			if (occupy_bus_lane) check_occupy_bus_lane(car, width, height);
		}

		Sleep(*video_info->detect_delay);//��ͣ
	}

	return 0;
}

#define CHECK_TICK(last_tick,value) if(++last_tick < value) return; else last_tick = 0;

void calc_human_traffic(int value)
{
	//2����һ��
	static int last_tick = 0;
	CHECK_TICK(last_tick, 30);

	//���õ�ǰ������
	g_global_set.secne_set.human_count.set_current_count(value);

	//��һ�ε�����
	static int last_count = 0;

	//��������
	g_global_set.secne_set.human_count.add_count(value - last_count);

	//��������
	last_count = value;
}

void calc_car_traffic(int value)
{
	//2����һ��
	static int last_tick = 0;
	CHECK_TICK(last_tick, 60);

	//���õ�ǰ������
	g_global_set.secne_set.car_count.set_current_count(value);

	//��һ�εĳ�����
	static int last_count = 0;

	//����������
	g_global_set.secne_set.car_count.add_count(value - last_count);

	//���浱ǰ������
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

	b.x = left;//��ʼλ�õ�x
	b.y = top;//��ʼλ�õ�y
	b.w = right;//����λ�õ�x
	b.h = bot;//����λ�õ�y
}

bool calc_intersect(box b1, box b2, float ratio)
{
	//����
	auto self_swap = [](box& _b1, box& _b2)
	{
		box temp = _b1;
		_b1 = _b2;
		_b2 = temp;
	};

	//�����ཻ�ж�
	if (b1.x > b2.x) self_swap(b1, b2);
	if (b1.w <= b2.x) return false;
	float radio_w = (b1.w - b2.x) / (b2.w - b1.x);

	//�����ཻ�ж�
	if (b1.y > b2.y) self_swap(b1, b2);
	if (b1.h <= b2.y) return false;
	float radio_h = (b1.h - b2.y) / (b2.h - b1.y);

	//��һ���ཻ�ʴ��ھ��ж�Ϊͬһ����
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
	//2����һ��
	static int last_tick = 0; 
	CHECK_TICK(last_tick, 30);

	//�빫�������ཻ�ĳ���
	std::vector<box> regions;

	//��������
	for (auto& it : g_global_set.mask_list)
	{
		//�ǹ�������
		if(it.type != region_bus_lane) continue;

		//������������
		box bus_region = it.get_box();
		calc_trust_box(bus_region, width, height);

		//��������
		for (auto& ls : b) calc_trust_box(ls, width, height);
		for (auto& ls : b)	if (calc_intersect(bus_region, ls)) regions.push_back(ls);
	}

	//��һ֡����ĳ���
	static std::vector<box> last_regions;

	for (const auto& it : regions)
	{
		//�ж���һ������һ֡�ǲ��Ǳ�������
		bool state = false;
		for (int i = 0; i < last_regions.size(); i++)
		{
			if (calc_intersect(it, last_regions[i]))//�ཻ����0.5���Ǳ�������
			{
				last_regions.erase(last_regions.begin() + i);
				state = true;
				break;
			}
		}
		if (!state)//û�б�����
		{
			scene_info::occupy_bus_info::bus_data temp;

			//����ʶ��....

			g_global_set.secne_set.bus_datas.push_bus_data(temp);
		}
	}

	//����
	last_regions = std::move(regions);

	/*

	//��һ�γ���λ��
	static std::vector<box> last_pos;

	//ȫ��ת��Ϊ��ʵλ��
	for (auto& it : b) calc_trust_box(it, width, height);

	//����ÿһ������
	for (auto& it : g_global_set.mask_list)
	{
		//����ǹ�����������
		if (it.type == region_bus_lane)
		{
			//����������ʵλ��
			box region_box = it.get_box();
			calc_trust_box(region_box, width, height);

			//�����빫�������Ƿ��ཻ
			for (int i = 0; i < b.size(); i++)
			{
				//�빫�������ཻ
				if (calc_intersect(region_box, b[i]))
				{
					//����ͬһ��
					if (!calc_same_rect(last_pos, b[i]))
					{
						car_info info;
						//���Ƽ��
						//........

						//��ȡ��ǰʱ��
						time_t timep;
						time(&timep);
						struct tm *prt = gmtime(&timep);
						info.times[0] = prt->tm_year + 1900;//��
						info.times[1] = prt->tm_mon + 1;//��
						info.times[2] = prt->tm_mday;//��
						info.times[3] = prt->tm_hour + 8;//ʱ
						info.times[4] = prt->tm_min;//��
						info.times[5] = prt->tm_sec;//��
						g_global_set.secne_set.occupy_bus_list.push_back(std::move(info));
					}
				}
				else//���ཻ ɾ��֮
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

//��ͼƬת��Ϊ��ǩ
void picture_to_label(const char* path, std::map<std::string, int>& class_names)
{
	//��ǩ�ļ�����
	int names_size = 0;

	//��ȡ��ǩ����
	char **names = get_labels_custom("h:\\test\\coco.names", &names_size); //get_labels(name_list);

	//��ȡ��������
	network net = parse_network_cfg_custom("h:\\test\\yolov3.cfg", 1, 1); // set batch=1

	//����Ȩ���ļ�
	load_weights(&net, "h:\\test\\yolov3.weights");

	//�ںϾ��
	fuse_conv_batchnorm(net);

	//���������Ȩ��
	calculate_binary_weights(net);

	//��ȡͼƬ�ļ�
	std::vector<std::string> picture_list = get_path_from_str(path, "*.jpg");

	printf("ͼƬ������ %d \n", picture_list.size());

	float thresh = 0.5f;
	float hier_thresh = 0.5f;
	float nms = 0.5f;

	//����ÿһ��ͼƬ
	for (int i = 0; i < picture_list.size(); i++)
	{
		if (i && i % 500 == 0) printf("ͼƬ���� %d    ������� : %d \n", picture_list.size(), i);

		//��������
		std::string jpg_name = path;
		jpg_name += "\\" + picture_list[i];

		//����ͼ���ת����С
		image im = load_image((char*)jpg_name.c_str(), 0, 0, net.c);
		if(im.data == NULL) continue;
		image sized = resize_image(im, net.w, net.h);

		//����Ԥ��
		network_predict(net, sized.data);

		//��ȡ������Ϣ
		int nboxes = 0;
		detection *dets = get_network_boxes(&net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, 0);

		//�Ǽ���ֵ����
		do_nms_sort(dets, nboxes, net.layers[net.n - 1].classes, nms);

		//��ȡ��Ч����
		int detections_num;
		detection_with_class* selected_detections = get_actual_detections(dets, nboxes, thresh, &detections_num, names);

		//�ж���ʹ����ļ�
		if (detections_num)
		{
			//�����ļ�����
			std::string file_name = jpg_name.substr(0, jpg_name.rfind('.'));
			file_name += ".txt";

			int useful_count = 0;

			//д��λ��
			std::fstream file(file_name, std::fstream::out | std::fstream::trunc);//ע���޸�����
			if (file.is_open())
			{
				//��ÿһ������
				for (int j = 0; j < detections_num; j++)
				{
					//��ȡ������
					int class_index = selected_detections[j].best_class;

					//��������һ��
					for (auto& it : class_names)
					{
						if (it.first == names[class_index])
						{
							//��ȡλ��
							box b = selected_detections[j].det.bbox;

							//д����Ϣ
							char format[1024];
							sprintf(format, "%d %f %f %f %f\n", it.second, b.x, b.y, b.w, b.h);//ע���޸�����
							file.write(format, strlen(format));

							//����
							useful_count++;

							//�˳���ǰѭ��
							break;
						}
					}
				}
			}

			//�ر��ļ�
			file.close();

			//û�ж����ɾ��txt�ļ�
			if (useful_count == 0) DeleteFileA(file_name.c_str());//��ס�޸�����
		}

		//�ͷ��ڴ�
		free_detections(dets, nboxes);
		free_image(im);
		free_image(sized);
		free(selected_detections);
	}

	//�ͷ�����
	free_ptrs((void**)names, names_size);

	//�ͷ�����
	free_network(net);
}
