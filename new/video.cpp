#include "video.h"

video* video::m_static_this = nullptr;

void __cdecl video::read_frame_thread(void* data)
{
	const auto mode = m_static_this->get_mode();
	auto capture = m_static_this->get_capture();
	auto frames_list = m_static_this->get_frames();

	if (mode == e_mode_video) capture->open(m_static_this->get_path());
	if (mode == e_mode_camera) capture->open(m_static_this->get_index());
	if (capture->isOpened() == false)
	{
		check_warning(false, "视频文件打开失败");
		return;
	}

	m_static_this->set_reading(true);
	while (true)
	{
		if(m_static_this->get_is_reading() == false) break;

		if (m_static_this->get_pause_state())
		{
			wait_time(1000);
			continue;
		}

		int count = frames_list->size();
		if (count < 10)
		{
			struct frame_handle* temp = new frame_handle;
			temp->state = e_un_handle;

			m_static_this->entry_capture_mutex();
			bool state = capture->read(temp->frame);
			m_static_this->leave_capture_mutex();

			if (state == false)
			{
				delete temp;
				break;
			}

			m_static_this->entry_frame_mutex();
			frames_list->push_back(temp);
			m_static_this->leave_frame_mutex();

		}

		wait_time(10);
	}

	capture->release();
	m_static_this->set_reading(false);
	m_static_this->close();
}

void __cdecl video::detect_frame_thread(void* data)
{
	auto frames_list = m_static_this->get_frames();

	m_static_this->set_detecting(true);
	while (true)
	{
		if (m_static_this->get_is_reading() == false) break;
		if (m_static_this->get_is_detecting() == false) break;

		if (m_static_this->get_pause_state())
		{
			wait_time(1000);
			continue;
		}

		struct frame_handle* temp = nullptr;

		m_static_this->entry_frame_mutex();
		for (auto& it = frames_list->begin(); it != frames_list->end(); it++)
		{
			if ((*it)->state == e_un_handle)
			{
				(*it)->state = e_detec_handle;
				temp = (*it);
				break;
			}
		}
		m_static_this->leave_frame_mutex();

		if (temp)
		{
			//检测.....
			temp->state = e_finish_handle;

		}

		wait_time(10);
	}

	m_static_this->set_detecting(false);
}

void video::update_fps() noexcept
{
	static double before = 0;

	double after = get_time_point();
	double current = 1000000.0 / (after - before);
	m_display_fps = m_display_fps * 0.9 + current * 0.1;
	before = after;
}

bool video::get_is_reading() const noexcept
{
	return m_reading;
}

bool video::get_is_detecting() const noexcept
{
	return m_detecting;
}

void video::set_reading(bool state) noexcept
{
	m_reading = state;
}

void video::set_detecting(bool state) noexcept
{
	m_detecting = state;
}

const char* video::get_path() const noexcept
{
	return m_path;
}

int video::get_index() const noexcept
{
	return m_index;
}

video_display_mode video::get_mode() const noexcept
{
	return m_mode;
}

cv::VideoCapture* video::get_capture() noexcept
{
	return &m_capture;
}

std::list<frame_handle*>* video::get_frames() noexcept
{
	return &m_frames;
}

void video::entry_capture_mutex() noexcept
{
	m_capture_mutex.lock();
}

void video::leave_capture_mutex() noexcept
{
	m_capture_mutex.unlock();
}

void video::entry_frame_mutex() noexcept
{
	m_frame_mutex.lock();
}

void video::leave_frame_mutex() noexcept
{
	m_frame_mutex.unlock();
}

bool video::get_pause_state() const noexcept
{
	return m_pause_video;
}

double video::get_display_fps() const noexcept
{
	return m_display_fps;
}

bool video::set_video_path(const char* path) noexcept
{
	if (m_path)
	{
		free_memory(m_path);
		m_path = nullptr;
	}

	r_size_t size = strlen(path);
	m_path = alloc_memory<char*>(size);
	if (m_path == nullptr) return false;
	strncpy(m_path, path, size);

	m_mode = e_mode_video;
	return true;
}

bool video::set_video_index(int index) noexcept
{
	m_index = index;

	m_mode = e_mode_camera;
	return true;
}

struct frame_handle* video::get_video_frame() noexcept
{
	struct frame_handle* temp = nullptr;

	entry_frame_mutex();
	for (auto& it = m_frames.begin(); it != m_frames.end(); it++)
	{
		if ((*it)->state == e_finish_handle)
		{
			this->update_fps();
			temp = (*it);
			m_frames.erase(it);
			break;
		}
	}
	leave_frame_mutex();

	return temp;
}

void video::set_frame_index(int index) noexcept
{
	if (m_capture.isOpened() == false) return;

	entry_capture_mutex();
	m_capture.set(cv::CAP_PROP_POS_FRAMES, index);
	leave_capture_mutex();
}

void video::set_frame_index(float rate) noexcept
{
	if (m_capture.isOpened() == false) return;

	float total = m_capture.get(cv::CAP_PROP_FRAME_COUNT);

	entry_capture_mutex();
	m_capture.set(cv::CAP_PROP_POS_FRAMES, rate * total);
	leave_capture_mutex();
}

float video::get_finish_rate() noexcept
{
	if (m_capture.isOpened() == false) return 0.0f;

	float total = m_capture.get(cv::CAP_PROP_FRAME_COUNT);
	float current = m_capture.get(cv::CAP_PROP_POS_FRAMES);
	return current / total;
}

video::video()
{
	m_static_this = this;

	m_reading = false;
	m_detecting = false;

	m_pause_video = false;

	m_display_fps = 0.0;
}

video::~video()
{
	if(m_path) free_memory(m_path);
}

bool video::start() noexcept
{
	if (m_path == nullptr) return false;

	this->close();

	auto func = [](_beginthread_proc_type ptr)
	{
		return _beginthread(ptr, 0, nullptr);
	};

	check_warning(func(read_frame_thread) != -1, "读取视频帧线程失败");
	wait_time(200);
	for (int i = 0; i < m_detect_count; i++)
		check_warning(func(detect_frame_thread) != -1, "检测视频帧线程失败");
	return true;
}

void video::pause() noexcept
{
	if(m_reading && m_detecting)
		m_pause_video = true;
}

void video::restart() noexcept
{
	if (m_reading && m_detecting)
		m_pause_video = false;
}

void video::close() noexcept
{
	m_reading = false;
	m_detecting = false;
	m_pause_video = false;
	wait_time(1000);
}
