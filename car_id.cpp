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

//��ȡ��������
data read_car_id_data(const char* images_path, int image_size, int classes);

//ѵ��ģ��
void train_model(const char* cfg_path, const char* weights_path, const char* classes_path, const char* image_path);


//
//int main(int argc,char* argv[])
//{
//
//	return 0;
//}

void train_model(const char* cfg_path, const char* weights_path, const char* classes_path, const char* image_path)
{
	//�����ļ�������
	if (!cfg_path || access(cfg_path, 0) == -1) return;
	if (!weights_path || access(weights_path, 0) == -1) return;
	if (!image_path || access(image_path, 0) == -1) return;

	//��������
	network net = parse_network_cfg((char*)cfg_path);

	//����Ȩ��
	load_weights(&net, (char*)weights_path);

	//��ȡ��ǩ�ļ�
	int classes;
	char** class_name = get_labels_custom((char*)classes_path, &classes);

	printf("ѧϰ��:%f\t ����:%f\t ˥��:%f\n", net.learning_rate, net.momentum, net.decay);

	//Ȩ���ļ�����Ŀ¼
	const char* backup = "backup";
	CreateDirectoryA(backup, NULL);

	//��ȡѧϰ����
	data train = read_car_id_data(image_path, 32 * 32 * 3, classes);

	//ƽ����ʧ
	float avg_loss = -1;

	//��ʼ����ѵ��
	while (get_current_batch(net) < net.max_batches)
	{
		//��ȡ��ʼʱ��
		clock_t this_time = clock();

		//��ʼѵ����ȡ��ʧ
		float loss = train_network_sgd(net, train, 1);

		//����ƽ����ʧ
		if (avg_loss == -1) avg_loss = loss;
		avg_loss = avg_loss * .95f + loss * .05f;

		//��ʾ��Ϣ
		printf("��������:%d  ѵ����ʧ:%f  ƽ����ʧ:%f  ��ǰѧϰ��:%f  ��ʱ:%f\n",
			get_current_batch(net), loss, avg_loss, get_current_rate(net), sec(clock() - this_time));

		//ÿ100�α���һ��Ȩ���ļ�
		if (get_current_batch(net) % 100 == 0)
		{
			char temp[1024];
			sprintf(temp, "%s\\%d.weights", backup, get_current_batch(net));
			save_weights(net, temp);
		}
	}

	//�˳�ǰҲҪ����һ��Ȩ���ļ�
	char temp[1024];
	sprintf(temp, "%s\\finish_%d.weights", backup, get_current_batch(net));
	save_weights(net, temp);

	//�ͷű�ǩ
	free_ptrs((void**)class_name, classes);

	//�ͷ�����
	free_network(net);

	//�ͷ�ѧϰ�ļ�
	free_data(train);

	//��ͣһ��
	system("pause");
}

data read_car_id_data(const char* images_path, int image_size, int classes)
{
	//����ȫ��ͼƬ·��
	std::vector<std::string> all_file;

	//����ȫ��ͼƬ��ǩ
	std::vector<int> all_label;

	//���ļ�
	std::fstream file(images_path, std::fstream::in);
	if (file.is_open() == false)
	{
		printf("%s �ļ���ʧ��\n", images_path);
		getchar();
		exit(-1);
	}

	//��ȡȫ��ͼƬ·���ͱ�ǩ
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

	//�����ռ�
	data result_data;
	result_data.shallow = 0;
	matrix X = make_matrix(all_file.size(), image_size);
	matrix y = make_matrix(all_file.size(), classes);
	result_data.X = X;
	result_data.y = y;

	//��ȡÿһ��ͼƬ
	for (int i = 0; i < all_file.size(); i++)
	{
		//��ȡͼƬ����
		cv::Mat src = cv::imread(all_file[i]);
		if(src.empty()) continue;
		unsigned char* this_data = (unsigned char*)src.data;

		//���ñ�ǩ
		y.vals[i][all_label[i]] = 1;

		//����ͼƬ����
		for (int j = 0; j < X.cols; j++) X.vals[i][j] = (double)this_data[j];
	}

	//ȫ����һ��
	scale_data_rows(result_data, 1.0f / 255.0f);

	//����
	return result_data;
}








