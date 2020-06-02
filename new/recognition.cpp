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
}

bool object_recognition::set_model_path(const char* path) noexcept
{
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

	//m_net = parse_network_cfg_custom(cfgfile, 1, 0);

	m_initialize_model = true;
	return true;
}