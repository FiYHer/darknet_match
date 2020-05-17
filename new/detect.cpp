#include "detect.h"

#include <fstream>
#include <string>
#include <vector>


object_detect::object_detect()
{
	m_path = nullptr;
	m_classes = nullptr;
	m_loaded = false;
	m_classes_count = 0;

	m_thresh = 0.25f;
	m_hier_thresh = 0.5f;
	m_nms = 0.45f;
}

object_detect::~object_detect()
{
	if (m_path) free_memory(m_path);
}

bool object_detect::load_model() noexcept
{
	if (m_loaded || m_path == nullptr) 
		return false;

	std::fstream file(m_path, std::fstream::in);
	if (file.is_open() == false) 
		return false;

	std::string names_path, cfg_path, weights_path;
	getline(file, names_path);
	getline(file, cfg_path);
	getline(file, weights_path);
	file.close();

	if (names_path.empty() || cfg_path.empty() || weights_path.empty())
		return false;

	file.open(names_path, std::fstream::in);
	if (file.is_open() == false)
		return false;

	std::vector<std::string> classes_buffer;
	std::string buffer;
	while (getline(file, buffer))
		classes_buffer.push_back(std::move(buffer));
	file.close();

	if (m_classes)
	{
		for (int i = 0; i < m_classes_count; i++)
			free_memory(m_classes[i]);
		free_memory(m_classes);
	}

	{
		int size = classes_buffer.size();
		m_classes_count = size;
		m_classes = alloc_memory<char**>(size);
		if (m_classes == nullptr)
			return false;

		for (int i = 0; i < size; i++)
		{
			int len = classes_buffer[i].size();
			m_classes[i] = alloc_memory<char*>(len);
			if (m_classes[i] == nullptr) return false;
			strncpy(m_classes[i], classes_buffer[i].c_str(), len);
		}
	}

	m_net = parse_network_cfg_custom((char*)cfg_path.c_str(), 1, 1);
	load_weights(&m_net, (char*)weights_path.c_str());
	fuse_conv_batchnorm(m_net);
	calculate_binary_weights(m_net);

	m_loaded = true;
	return true;
}

bool object_detect::unload_model() noexcept
{
	if (m_loaded)
		free_network(m_net);

	return true;
}

bool object_detect::set_model_path(const char* path) noexcept
{
	if(m_path) free_memory(m_path);

	int size = strlen(path);
	m_path = alloc_memory<char*>(size);
	if (m_path == nullptr) return false;

	strncpy(m_path, path, size);
	return true;
}

network* object_detect::get_network() noexcept
{
	return &m_net;
}

int object_detect::get_classes_count() const noexcept
{
	return m_classes_count;
}

bool object_detect::get_model_loader() const noexcept
{
	return m_loaded;
}

float object_detect::get_thresh() const noexcept
{
	return m_thresh;
}

float object_detect::get_hier_thresh() const noexcept
{
	return m_hier_thresh;
}

float object_detect::get_nms() const noexcept
{
	return m_nms;
}
