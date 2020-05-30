#include "help.h"

void do_handle(bool is_exit, const char* format...) noexcept
{
	char buffer[max_string_len]{ 0 };
	va_list format_list;
	va_start(format_list, format);
	vsprintf_s(buffer, format, format_list);
	va_end(format_list);

	MessageBoxA(0, buffer, "information", 0);
	if (is_exit) exit(-1);
}

void check_error(bool state, const char* format...) noexcept
{
	if (state) return;
	do_handle(true, format);
}

void check_warning(bool state, const char* format...) noexcept
{
	if (state) return;
	do_handle(false, format);
}

void wait_time(int i, bool b) noexcept
{
	if (b) Sleep(i);
	else
	{
#ifdef _DEBUG
		Sleep(i);
#endif
	}
}

void free_memory(void* data)
{
	if (data) VirtualFree(data, 0, MEM_RELEASE);
	data = nullptr;
}

char* get_file_path(const char* file_type, char* buffer, int size) noexcept
{
	memset(buffer, 0, sizeof(char) * size);
	OPENFILENAMEA info{ 0 };
	info.lStructSize = sizeof(info);
	info.lpstrFilter = file_type;
	info.nFilterIndex = 1;
	info.lpstrFile = buffer;
	info.nMaxFile = size;
	info.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_HIDEREADONLY;
	GetOpenFileNameA(&info);
	return buffer;
}

int get_file_type(const char* path) noexcept
{
	int len = strlen(path);
	if (len > 3)
	{
		//视频文件
		if (path[len - 3] == 'm' && path[len - 2] == 'p' && path[len - 1] == '4') return 1;
		if (path[len - 3] == 'f' && path[len - 2] == 'l' && path[len - 1] == 'v') return 1;
		if (path[len - 3] == 'a' && path[len - 2] == 'v' && path[len - 1] == 'i') return 1;
		if (path[len - 2] == 't' && path[len - 1] == 's') return 1;

		//图片文件
		if (path[len - 3] == 'j' && path[len - 2] == 'p' && path[len - 1] == 'g') return 2;
		if (path[len - 3] == 'p' && path[len - 2] == 'n' && path[len - 1] == 'g') return 2;
		if (path[len - 3] == 'b' && path[len - 2] == 'm' && path[len - 1] == 'p') return 2;
	}
	return 0;
}

int get_current_minute() noexcept
{
	time_t t = time(0);
	return localtime(&t)->tm_min;
}