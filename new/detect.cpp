#include "detect.h"

#include <fstream>
#include <string>
#include <vector>

#include <time.h>

object_detect::object_detect()
{
	m_path = nullptr;
	m_classes_name = nullptr;
	m_loaded = false;
	m_classes_count = 0;

	m_classes_color = nullptr;

	m_thresh = 0.25f;
	m_hier_thresh = 0.5f;
	m_nms = 0.45f;
}

object_detect::~object_detect()
{
	if (m_path) free_memory(m_path);
	free_classes_name();
	free_classes_color();
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

	if (m_classes_name)
	{
		for (int i = 0; i < m_classes_count; i++)
			free_memory(m_classes_name[i]);
		free_memory(m_classes_name);
	}

	{
		int size = classes_buffer.size();
		m_classes_count = size;
		m_classes_name = alloc_memory<char**>(size);
		if (m_classes_name == nullptr)
			return false;

		for (int i = 0; i < size; i++)
		{
			int len = classes_buffer[i].size();
			m_classes_name[i] = alloc_memory<char*>(len);
			if (m_classes_name[i] == nullptr) return false;
			strncpy(m_classes_name[i], classes_buffer[i].c_str(), len);
		}
	}

	{
		srand((unsigned int)time(nullptr));

		int size = classes_buffer.size();
		m_classes_color = alloc_memory<float**>(size);
		if (m_classes_color == nullptr)
			return false;

		for (int i = 0; i < size; i++)
		{
			m_classes_color[i] = alloc_memory<float*>(3);
			if (m_classes_color[i] == nullptr)
				return false;
			m_classes_color[i][0] = rand() % 255;
			m_classes_color[i][1] = rand() % 255;
			m_classes_color[i][2] = rand() % 255;
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
	free_classes_name();
	free_classes_color();
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

char** object_detect::get_classes_name() const noexcept
{
	return m_classes_name;
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

float** object_detect::get_classes_color() const noexcept
{
	return m_classes_color;
}

void object_detect::free_classes_name() noexcept
{
	if (m_classes_name)
	{
		for (int i = 0; i < m_classes_count; i++)
			free_memory(m_classes_name[i]);
		free_memory(m_classes_name);
	}
	m_classes_name = nullptr;
	m_classes_count = 0;
}

void object_detect::free_classes_color() noexcept
{

}
