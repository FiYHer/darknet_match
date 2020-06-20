#include "recognition.h"

object_recognition::object_recognition() noexcept
{
	m_initialize_model = false;

	m_path[0] = '\x0';

	m_labels_name = nullptr;
	m_labels_count = 0;
}

object_recognition::~object_recognition() noexcept
{
	unload_model();
}

bool object_recognition::set_model_path(const char* path) noexcept
{
	//字符串有效性判断
	int size = strlen(path);
	if (size <= 0 || size > max_string_len) return false;

	strncpy_s(m_path, path, size);
	return true;
}

bool object_recognition::load_model() noexcept
{
	if (m_initialize_model)
	{
		check_warning(false, "识别模型不能二次加载");
		return false;
	}

	std::fstream file(m_path, std::fstream::in);
	if (file.is_open() == false)
	{
		check_warning(false, "模型文件打开失败");
		return false;
	}

	std::string names_file, cfg_file, weights_file;
	getline(file, names_file);
	getline(file, cfg_file);
	getline(file, weights_file);
	file.close();

	if (names_file.empty() || cfg_file.empty() || weights_file.empty())
	{
		check_warning(false, "模型文件内容错误");
		return false;
	}

	file.open(names_file, std::fstream::in);
	if (file.is_open() == false)
	{
		check_warning(false, "names文件有误");
		return false;
	}

	std::vector<std::string> names_list;
	std::string line_buffer;
	while (getline(file, line_buffer)) names_list.push_back(std::move(line_buffer));
	file.close();

	m_labels_count = names_list.size();
	m_labels_name = new char*[m_labels_count];
	for (int i = 0; i < m_labels_count; i++)
	{
		int len = names_list[i].size();
		m_labels_name[i] = new char[len];
		strncpy(m_labels_name[i], names_list[i].c_str(), len);
	}

	//加载网络模型
	m_net = parse_network_cfg_custom((char*)cfg_file.c_str(), 1, 0);
	load_weights(&m_net, (char*)weights_file.c_str());
	set_batch_network(&m_net, 1);
	fuse_conv_batchnorm(m_net);
	calculate_binary_weights(m_net);

	m_initialize_model = true;
	return true;
}

bool object_recognition::unload_model() noexcept
{
	if (m_initialize_model)
		free_network(m_net);

	free_lables();

	m_initialize_model = false;
	return true;
}

bool object_recognition::is_loaded() noexcept
{
	return m_initialize_model;
}

void object_recognition::free_lables() noexcept
{
	if (m_labels_name)
	{
		for (int i = 0; i < m_labels_count; i++) delete[] m_labels_name[i];
		delete[] m_labels_name;
	}

	m_labels_name = nullptr;
	m_labels_count = 0;
}

void object_recognition::analyse(cv::Mat* roi, int results[7]) noexcept
{
	if (roi->empty()) return;

	//车牌一共7个字符
	for (int i = 0; i < 7; i++) results[i] = -1;

	int width = roi->cols;
	int height = roi->rows;
	int per = width * 3 / 9 / 2;

	//识别车牌的前两个字符
	results[0] = get_per_car_id((*roi)({ 0,0,per,height }));
	results[1] = get_per_car_id((*roi)({ per,0,per,height }));

	int start = 2 * per;
	per = (width - start) / 5;

	//设备车牌的后五个字符
	for (int i = 0; i < 5; i++) results[i + 2] = get_per_car_id((*roi)({ start + i * per,0,per,height }));
}

int object_recognition::get_per_car_id(cv::Mat buffer) noexcept
{
	//将车牌字符图像格式转化后resize
	image im = to_image(buffer);
	image resized = resize_min(im, m_net.w);
	image r = crop_image(resized, (resized.w - m_net.w) / 2, (resized.h - m_net.h) / 2, m_net.w, m_net.h);

	//预测车牌和排序
	float *predictions = network_predict(m_net, r.data);
	int indexes[3] = { 0,0,0 };
	top_k(predictions, m_net.outputs, 3, indexes);

	//释放相关内存
	if (r.data != im.data) free_image(r);
	free_image(im);
	free_image(resized);

	//返回索引
	return (int)predictions[indexes[0]];
}

image object_recognition::to_image(cv::Mat mat) noexcept
{
	int w = mat.cols;
	int h = mat.rows;
	int c = mat.channels();
	image im = make_image(w, h, c);
	unsigned char *data = (unsigned char *)mat.data;
	int step = mat.step;
	for (int y = 0; y < h; ++y) {
		for (int k = 0; k < c; ++k) {
			for (int x = 0; x < w; ++x) {
				im.data[k*w*h + y * w + x] = data[y*step + x * c + k] / 255.0f;
			}
		}
	}
	return im;
}