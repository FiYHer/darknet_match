
#include "gui.h"

int __stdcall WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd)
//int main(int argc, char* argv[])
{
	gui g;
	g.create_and_show();
	g.msg_handle();
	return 0;
}




