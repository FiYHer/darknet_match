#include "network.h"
#include "utils.h"
#include "parser.h"
#include "option_list.h"
#include "blas.h"
#include "darkunistd.h"

#include <vector>
#include <string>
#include <fstream>

#include <opencv2/opencv.hpp>

//读取车牌数据
data read_car_id_data(const char* images_path, int image_size, int classes);

//训练模型
void train_model(const char* cfg_path, const char* weights_path, const char* classes_path, const char* image_path);


//
//int main(int argc,char* argv[])
//{
//
//	return 0;
//}

void train_model(const char* cfg_path, const char* weights_path, const char* classes_path, const char* image_path)
{
	//检验文件合理性
	if (!cfg_path || access(cfg_path, 0) == -1) return;
	if (!weights_path || access(weights_path, 0) == -1) return;
	if (!image_path || access(image_path, 0) == -1) return;

	//加载网络
	network net = parse_network_cfg((char*)cfg_path);

	//加载权重
	load_weights(&net, (char*)weights_path);

	//读取标签文件
	int classes;
	char** class_name = get_labels_custom((char*)classes_path, &classes);

	printf("学习率:%f\t 动量:%f\t 衰减:%f\n", net.learning_rate, net.momentum, net.decay);

	//权重文件保存目录
	const char* backup = "backup";
	CreateDirectoryA(backup, NULL);

	//读取学习数据
	data train = read_car_id_data(image_path, 32 * 32 * 3, classes);

	//平均损失
	float avg_loss = -1;

	//开始迭代训练
	while (get_current_batch(net) < net.max_batches)
	{
		//获取开始时间
		clock_t this_time = clock();

		//开始训练获取损失
		float loss = train_network_sgd(net, train, 1);

		//计算平均损失
		if (avg_loss == -1) avg_loss = loss;
		avg_loss = avg_loss * .95f + loss * .05f;

		//显示信息
		printf("迭代次数:%d  训练损失:%f  平均损失:%f  当前学习率:%f  耗时:%f\n",
			get_current_batch(net), loss, avg_loss, get_current_rate(net), sec(clock() - this_time));

		//每100次保存一次权重文件
		if (get_current_batch(net) % 100 == 0)
		{
			char temp[1024];
			sprintf(temp, "%s\\%d.weights", backup, get_current_batch(net));
			save_weights(net, temp);
		}
	}

	//退出前也要保存一次权重文件
	char temp[1024];
	sprintf(temp, "%s\\finish_%d.weights", backup, get_current_batch(net));
	save_weights(net, temp);

	//释放标签
	free_ptrs((void**)class_name, classes);

	//释放网络
	free_network(net);

	//释放学习文件
	free_data(train);

	//暂停一下
	system("pause");
}

data read_car_id_data(const char* images_path, int image_size, int classes)
{
	//保存全部图片路径
	std::vector<std::string> all_file;

	//保存全部图片标签
	std::vector<int> all_label;

	//打开文件
	std::fstream file(images_path, std::fstream::in);
	if (file.is_open() == false)
	{
		printf("%s 文件打开失败\n", images_path);
		getchar();
		exit(-1);
	}

	//读取全部图片路径和标签
	std::string line_data;
	while (getline(file, line_data))
	{
		char this_name[1024];
		int this_label;
		sscanf(line_data.c_str(), "%s - %d", this_name, &this_label);
		all_file.push_back(std::move(this_name));
		all_label.push_back(std::move(this_label));
	}
	file.close();

	//创建空间
	data result_data;
	result_data.shallow = 0;
	matrix X = make_matrix(all_file.size(), image_size);
	matrix y = make_matrix(all_file.size(), classes);
	result_data.X = X;
	result_data.y = y;

	//读取每一张图片
	for (int i = 0; i < all_file.size(); i++)
	{
		//读取图片数据
		cv::Mat src = cv::imread(all_file[i]);
		if(src.empty()) continue;
		unsigned char* this_data = (unsigned char*)src.data;

		//设置标签
		y.vals[i][all_label[i]] = 1;

		//设置图片数据
		for (int j = 0; j < X.cols; j++) X.vals[i][j] = (double)this_data[j];
	}

	//全部归一化
	scale_data_rows(result_data, 1.0f / 255.0f);

	//返回
	return result_data;
}








