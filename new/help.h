#pragma once

#include <Windows.h>
#include <stdio.h>

using r_hresult = HRESULT;
using r_hwnd = HWND;
using r_dword = DWORD;
using r_uint = UINT;
using r_wparam = WPARAM;
using r_lparam = LPARAM;
using r_size_t = SIZE_T;
using r_handle = HANDLE;

constexpr int max_string_len = 1024;

//处理
void do_handle(bool is_exit, const char* format...) noexcept;

//检测错误
void check_error(bool state, const char* format...) noexcept;

//发出警告
void check_warning(bool state, const char* format...) noexcept;

//等待时间
void wait_time(int i, bool b = false) noexcept;

//申请空间
template <class T>
T alloc_memory(r_size_t size)
{
	return (T)VirtualAlloc(nullptr, size, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
}

//释放空间
void free_memory(void* data);

//选择文件
char* get_file_path(const char* file_type, char* buffer, int size) noexcept;

//获取文件的类型
int get_file_type(const char* path) noexcept;
