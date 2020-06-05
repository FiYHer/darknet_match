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
		if (m_static_this->get_is_reading() == false) break;
		if (m_static_this->get_pause_state())
		{
			wait_time(500, true);
			continue;
		}

		m_static_this->entry_frame_mutex();
		int count = frames_list->size();
		m_static_this->leave_frame_mutex();

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

		wait_time(15, true);
	}

	capture->release();
	m_static_this->set_reading(false);
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
			wait_time(500, true);
			continue;
		}

		struct frame_handle* temp = nullptr;
		m_static_this->entry_frame_mutex();
		for (auto& it : *frames_list)
		{
			if (it->state == e_un_handle)
			{
				it->state = e_detec_handle;
				temp = it;
				break;
			}
		}
		m_static_this->leave_frame_mutex();

		if (temp)
		{
			//检测模型是否加载
			bool status = m_static_this->get_detect_model()->get_model_loader();
			if (status)
			{
				//转化视频帧
				network* net = m_static_this->get_detect_model()->get_network();
				int classes_count = m_static_this->get_detect_model()->get_classes_count();
				image mat = m_static_this->to_image(temp->frame, net->w, net->h, net->c);

				//网络预测
				network_predict(*net, mat.data);

				//阈值相关
				float thresh = m_static_this->get_detect_model()->get_thresh();
				float hier_thresh = m_static_this->get_detect_model()->get_hier_thresh();
				float nms = m_static_this->get_detect_model()->get_nms();

				//获取方框数量
				int box_count = 0;
				detection* result = get_network_boxes(net, net->w, net->h, thresh, hier_thresh, 0, 1, &box_count, 0);

				//非极大值抑制
				do_nms_sort(result, box_count, classes_count, nms);

				//相关场景检测
				m_static_this->scene_manager(result, box_count, temp->frame.cols, temp->frame.rows, &temp->frame);

				//绘制方框和字体
				m_static_this->draw_box_and_font(result, box_count, &temp->frame);

				//绘制区域
				m_static_this->draw_regions(&temp->frame);

				//释放内存
				free_image(mat);
				free_detections(result, box_count);
			}

			//完成标记
			temp->state = e_finish_handle;
		}
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

bool video::per_frame()
{
	bool state = m_capture.isOpened();
	if (state == false)
	{
		check_warning(false, "没有打开视频文件");
		return false;
	}

	struct frame_handle* temp = new frame_handle;
	temp->state = e_finish_handle;

	state = m_capture.read(temp->frame);
	if (state == false)
	{
		check_warning(false, "读取一张视频帧失败");
		delete temp;
		return false;
	}

	this->entry_frame_mutex();
	m_frames.push_back(temp);
	this->leave_frame_mutex();

	return true;
}

void video::box_to_pos(box b, int w, int h, int& left, int& top, int& right, int& bot)
{
	if (std::isnan(b.w) || std::isinf(b.w)) b.w = 0.5;
	if (std::isnan(b.h) || std::isinf(b.h)) b.h = 0.5;
	if (std::isnan(b.x) || std::isinf(b.x)) b.x = 0.5;
	if (std::isnan(b.y) || std::isinf(b.y)) b.y = 0.5;

	b.w = (b.w < 1) ? b.w : 1;
	b.h = (b.h < 1) ? b.h : 1;
	b.x = (b.x < 1) ? b.x : 1;
	b.y = (b.y < 1) ? b.y : 1;

	left = (b.x - b.w / 2.)*w;
	right = (b.x + b.w / 2.)*w;
	top = (b.y - b.h / 2.)*h;
	bot = (b.y + b.h / 2.)*h;

	if (left < 0) left = 0;
	if (right > w - 1) right = w - 1;
	if (top < 0) top = 0;
	if (bot > h - 1) bot = h - 1;
}

bool video::is_coincide_rate(box a, box b, float value /*= 0.5f*/)
{
	if (b.x < a.x)
	{
		box c = a;
		a = b;
		b = c;
	}

	{
		float _max = b.w - a.x;
		float _min = a.w - b.x;
		if (_max > 0 && _min > 0)
		{
			float temp = _min / _max;
			if (temp > value) return true;
		}
	}

	if (b.y < a.y)
	{
		box c = a;
		a = b;
		b = c;
	}

	{
		float _max = b.h - a.y;
		float _min = a.h - b.y;
		if (_max > 0 && _min > 0)
		{
			float temp = _min / _max;
			if (temp > value) return true;
		}
	}

	return false;
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

void video::set_payse_state() noexcept
{
	m_pause_video = true;
}

bool video::get_pause_state() const noexcept
{
	return m_pause_video;
}

double video::get_display_fps() const noexcept
{
	return m_display_fps;
}

object_detect* video::get_detect_model() noexcept
{
	return &m_detect_model;
}

image video::to_image(cv::Mat frame, int out_w, int out_h, int out_c) noexcept
{
	cv::Mat temp = cv::Mat(out_w, out_h, out_c);
	cv::resize(frame, temp, temp.size(), 0, 0, cv::INTER_LINEAR);
	if (out_c > 1) cv::cvtColor(temp, temp, cv::COLOR_RGB2BGR);

	image im = make_image(out_w, out_h, out_c);
	unsigned char *data = (unsigned char *)temp.data;
	int step = temp.step;
	for (int y = 0; y < out_h; ++y)
	{
		for (int k = 0; k < out_c; ++k)
		{
			for (int x = 0; x < out_w; ++x)
			{
				im.data[k*out_w*out_h + y * out_w + x] = data[y*step + x * out_c + k] / 255.0f;
			}
		}
	}
	return im;
}

void video::draw_box_and_font(detection* detect, int count, cv::Mat* frame) noexcept
{
	char buffer[max_string_len]{ 0 };
	char** classes_name = m_detect_model.get_classes_name();
	int classes_count = m_detect_model.get_classes_count();
	float thresh = m_detect_model.get_thresh();
	float** classes_color = m_detect_model.get_classes_color();

	for (int i = 0; i < count; i++)
	{
		for (int j = 0; j < classes_count; j++)
		{
			if (detect[i].prob[j] > thresh)
			{
				box b = detect[i].bbox;
				int left, top, right, bot;
				box_to_pos(b, frame->cols, frame->rows, left, top, right, bot);

				//画方框
				{
					cv::Scalar color{ classes_color[j][0],classes_color[j][1], classes_color[j][2] };
					cv::rectangle(*frame, { left,top }, { right,bot }, color, 2.0f, 8, 0);
				}

				//画文字
				sprintf_s(buffer, "%s %2.f%%", classes_name[j], detect[i].prob[j] * 100.0f);
				float const font_size = frame->rows / 700.0f;
				cv::Size const text_size = cv::getTextSize(buffer, cv::FONT_HERSHEY_COMPLEX_SMALL, font_size, 1, 0);

				cv::Point pt_left, pt_right;
				{
					pt_left.x = left;
					int temp = top - text_size.height * 2;
					if (temp < 0) temp = top;
					pt_left.y = temp;

					pt_right.x = right;
					pt_right.y = top;
				}

				{
					cv::Scalar color{ 255.0f - classes_color[j][0], 255.0f - classes_color[j][1], 255.0f - classes_color[j][2] };
					cv::rectangle(*frame, pt_left, pt_right, color, 1.0f, 8, 0);
					cv::rectangle(*frame, pt_left, pt_right, color, -1, 8, 0);
				}

				{
					cv::Scalar color{ classes_color[j][0],classes_color[j][1], classes_color[j][2] };
					cv::Point pos{ pt_left.x, pt_left.y + text_size.height };
					cv::putText(*frame, buffer, pos, cv::FONT_HERSHEY_COMPLEX_SMALL, font_size, color, 2 * font_size, 16);
				}
			}
		}
	}
}

void video::draw_regions(cv::Mat* frame) noexcept
{
	entry_region_mutex();

	for (auto& it : m_regions)
	{
		box b = it.to_box_data();
		int left = b.x *  frame->cols;
		int right = b.w *  frame->cols;
		int top = b.y *  frame->rows;
		int bot = b.h * frame->rows;

		cv::rectangle(*frame, { left,top }, { right,bot }, it.color, 2.0f, 8, 0);
	}

	leave_region_mutex();
}

std::vector<region_info> video::get_region_list() const noexcept
{
	return m_regions;
}

void video::entry_region_mutex() noexcept
{
	m_region_mutex.lock();
}

void video::leave_region_mutex() noexcept
{
	m_region_mutex.unlock();
}

void video::push_region_back(struct region_info& region) noexcept
{
	entry_region_mutex();
	m_regions.push_back(region);
	leave_region_mutex();
}

void video::pop_region_back() noexcept
{
	entry_region_mutex();
	if (m_regions.size()) m_regions.pop_back();
	leave_region_mutex();
}

void video::scene_manager(detection* detect, int count, int w, int h, cv::Mat* frame) noexcept
{
	static int last_frame = 30;
	if (++last_frame < get_display_fps()) return;
	last_frame = 0;

	int classes_count = m_detect_model.get_classes_count();
	float thresh = m_detect_model.get_thresh();

	using box_list = std::vector<box>;
	box_list license_plate_list;
	box_list car_list;
	box_list bus_list;
	box_list people_list;
	box_list traffic_light_list;

	for (int i = 0; i < count; i++)
	{
		for (int j = 0; j < classes_count; j++)
		{
			if (detect[i].prob[j] > thresh)
			{
				box b = detect[i].bbox;
				int left, top, right, bot;
				this->box_to_pos(b, w, h, left, top, right, bot);
				b.x = left;
				b.y = top;
				b.w = right;
				b.h = bot;

				switch (j)
				{
				case object_license_plate: license_plate_list.push_back(b); break;
				case object_car: car_list.push_back(b); break;
				case object_bus: bus_list.push_back(b); break;
				case objetc_person: people_list.push_back(b); break;
				case object_traffic_light: traffic_light_list.push_back(b); break;
				}
				break;
			}
		}
	}

	if (m_calc_people.enable) scene_calc_people(people_list, frame);
	if (m_calc_car.enable) scene_calc_car(car_list, frame);
}

void video::scene_calc_people(std::vector<box> b, cv::Mat* frame) noexcept
{
	using box_list = std::vector<box>;
	static box_list last_box;

	unsigned int size = b.size();
	unsigned int new_val = size;

	for (const auto& it : b)
	{
		bool status = true;

		for (auto ls = last_box.begin(); ls != last_box.end(); ls++)
		{
			if (is_coincide_rate(it, *ls, 0.2f))
			{
				last_box.erase(ls);
				new_val--;
				status = false;
				break;
			}
		}

		if (status)
		{
			cv::Rect r{ (int)it.x, (int)it.y, (int)it.w - (int)it.x, (int)it.h - (int)it.y };
			cv::Mat src = (*frame)(r);
			m_calc_people.update_image(src);
		}
	}

	m_calc_people.update_current_val(size);
	m_calc_people.update_new_val(new_val);

	last_box = std::move(b);
}

void video::scene_calc_car(std::vector<box> b, cv::Mat* frame) noexcept
{
	using box_list = std::vector<box>;
	static box_list last_box;

	unsigned int size = b.size();
	unsigned int new_val = size;

	for (const auto& it : b)
	{
		bool status = true;

		for (auto ls = last_box.begin(); ls != last_box.end(); ls++)
		{
			if (is_coincide_rate(it, *ls, 0.2f))
			{
				last_box.erase(ls);
				new_val--;
				status = false;
				break;
			}
		}

		if (status)
		{
			cv::Rect r{ (int)it.x, (int)it.y, (int)it.w - (int)it.x, (int)it.h - (int)it.y };
			cv::Mat src = (*frame)(r);
			m_calc_car.update_image(src);
		}
	}

	m_calc_car.update_current_val(size);
	m_calc_car.update_new_val(new_val);

	last_box = std::move(b);
}

void video::scene_occupy_bus(std::vector<box> b) noexcept
{
}

struct calc_statistics_info* video::get_people_info_point() noexcept
{
	return &m_calc_people;
}

struct calc_statistics_info* video::get_car_info_point() noexcept
{
	return &m_calc_car;
}

bool video::set_video_path(const char* path) noexcept
{
	if (get_file_type(path) != 1) return false;

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

void video::get_per_video_frame(const char* path)
{
	if (m_reading)
	{
		check_warning(false, "视频读取中");
		return;
	}

	bool state = m_capture.open(path);
	if (state == false)
	{
		check_warning(false, "打开视频文件 %s 失败", path);
		return;
	}

	state = per_frame();
	if (state == false)
	{
		check_warning(false, "读取视频帧失败");
		return;
	}
}

void video::get_per_video_frame(int index)
{
	if (m_detecting)
	{
		check_warning(false, "视频读取中");
		return;
	}

	bool state = m_capture.open(index);
	if (state == false)
	{
		check_warning(false, "打开视频文件 %d 失败", index);
		return;
	}

	state = per_frame();
	if (state == false)
	{
		check_warning(false, "读取视频帧失败");
		return;
	}
}

video::video()
{
	m_static_this = this;

	m_reading = false;
	m_detecting = false;
	m_pause_video = false;

	m_path = nullptr;
	m_index = 0;

	m_mode = e_mode_video;

	m_display_fps = 0;
}

video::~video()
{
	if (m_path) free_memory(m_path);
	this->close();
}

bool video::start() noexcept
{
	if (m_reading)
	{
		check_warning(false, "视频正在播放中");
		return false;
	}

	if (m_mode == e_mode_video)
	{
		if (m_path == nullptr)
		{
			check_warning(false, "视频路径有误");
			return false;
		}
	}

	this->close();

	auto func = [](_beginthread_proc_type ptr)
	{
		return _beginthread(ptr, 0, nullptr);
	};

	check_warning(func(read_frame_thread) != -1, "读取视频帧线程失败");
	wait_time(400, true);
	check_warning(func(detect_frame_thread) != -1, "检测视频帧线程失败");
	return true;
}

void video::pause() noexcept
{
	if (m_reading && m_detecting)
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
	wait_time(1000, true);

	entry_frame_mutex();
	for (auto& it : m_frames)
	{
		it->frame.release();
		delete it;
	}
	m_frames.clear();
	leave_frame_mutex();
}