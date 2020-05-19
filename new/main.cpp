
#include "gui.h"

//int __stdcall WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd)
int main(int argc, char* argv[])
{
	cuda_set_device(cuda_get_device());
	CHECK_CUDA(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));

	gui g;
	g.create_and_show();
	g.msg_handle();
	return 0;
}




