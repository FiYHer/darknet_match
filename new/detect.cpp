#include "detect.h"

#include <fstream>
#include <string>
#include <vector>


bool detection::load_model() noexcept
{
	if (m_path == nullptr) 
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

	return true;
}

bool detection::unload_model() noexcept
{
	if (m_net.batch)
		free_network(m_net);
	m_net.batch = 0;

	return true;
}

bool detection::set_model_path(const char* path) noexcept
{
	if(m_path) free_memory(m_path);

	int size = strlen(path);
	m_path = alloc_memory<char*>(size);
	if (m_path == nullptr) return false;

	strncpy(m_path, path, size);
	return true;
}
