#include "NNpredef.h"

class NN
{
public :
    void init();
    void start_train(int);
    void start_test();

private :
    float input_layer[L1];
    float hidden_layer[L2];
    float output_layer[L3];
    float stand_out[L3];
    float weight1[L2][L1];
    float weight2[L3][L2];
    float bias1[L2];
    float bias2[L3];

    float activation(float);
    void inference();
    int find_max();
    float rand_generate();   
    void train();
    void train_iteration();
    void train_epoch(int);



};

float NN::activation(float number)
{
    return 1/(1+exp(-number));
}

void NN::inference()
{
    for(int j=0;j<L2;j++)
    {
        hidden_layer[j]=0;
        for(int i=0;i<L1;i++)
        {
            hidden_layer[j]+=input_layer[i]*weight1[j][i];
        }
        hidden_layer[j]+=bias1[j];
        hidden_layer[j]=activation(hidden_layer[j]);
    }
    
    for(int k=0;k<L3;k++)
    {
        output_layer[k]=0;
        for(int j=0;j<L2;j++)
        {
            output_layer[k]+=hidden_layer[j]*weight2[k][j];
        }
        output_layer[k]+=bias2[k];
        output_layer[k]=activation(output_layer[k]);
    }
}

int NN::find_max()
{
    int location=0;
    float max=0;
    for(int k=0;k<L3;k++)
    {
        if(output_layer[k]>max)
        {
            location=k;
            max=output_layer[k];
        }
    }
    return location;   
}

float NN::rand_generate()
{
    float number=(float)rand();
    number=(number-RAND_MAX/2.0)/(RAND_MAX/2.0);
    return number;
}

void NN::init()
{
    for(int j=0;j<L2;j++)
    {
        for(int i=0;i<L1;i++)
        {
            weight1[j][i]=rand_generate();
        }
        bias1[j]=rand_generate();
        hidden_layer[j]=0;
    }
    for(int k=0;k<L3;k++)
    {
        for(int j=0;j<L2;j++)
        {
            weight2[k][j]=rand_generate();
        }
        bias2[k]=rand_generate();
        output_layer[k]=0;
        stand_out[k]=0;
    }
}

void NN::train()
{
    inference();
    float delta3[L3]={0};
    float delta2[L2]={0};
    for(int k=0;k<L3;k++)
    {
        delta3[k]=(output_layer[k]-stand_out[k])*output_layer[k]*(1-output_layer[k]);   
        bias2[k]+=-yita*delta3[k];
    }
    for(int k=0;k<L3;k++)
    {
        for(int j=0;j<L2;j++)
        {
            weight2[k][j]+=-yita*hidden_layer[j]*delta3[k]; 
        }
    }
    for(int j=0;j<L2;j++)
    {
        delta2[j]=0;
        for(int k=0;k<L3;k++)
        {
            delta2[j]+=weight2[k][j]*delta3[k]*hidden_layer[j]*(1-hidden_layer[j]);
        }
        bias1[j]+=-yita*delta2[j];
    }
    for(int j=0;j<L2;j++)
    {
        for(int i=0;i<L1;i++)
        {
            weight1[j][i]+=-yita*input_layer[i]*delta2[j];
        }
    }
}

void NN::train_iteration()
{
    FILE *train_image=fopen("../train-images.idx3-ubyte","rb");
    FILE *train_label=fopen("../train-labels.idx1-ubyte","rb");
    
    uint image_header[4];
    uint label_header[2];
    fread(&image_header[0],sizeof(uint),4,train_image);
    fread(&label_header[0],sizeof(uint),2,train_label);

    uchar image_data[L1];
    uchar label_data;

    for(int var=0;var<train_count;var++)
    {
        fread(&label_data,sizeof(uchar),1,train_label);
        int index=(int)label_data;
        stand_out[index]=1;
        fread(&image_data[0],sizeof(uchar),L1,train_image);
        for(int i=0;i<L1;i++)
        {
            // input_layer[i]=((float)image_data[i])/255.0;
            if((int)image_data[i]>127)input_layer[i]=1;
            else input_layer[i]=0;
        }
        train();
        stand_out[index]=0;
    }
}

void NN::train_epoch(int epoch)
{
    for(int i=0;i<epoch;i++)
    {
        train_iteration();
    }
}

void NN::start_train(int epoch)
{
    train_epoch(epoch);
}

void NN::start_test()
{
    FILE *test_image=fopen("../t10k-images.idx3-ubyte","rb");
    FILE *test_label=fopen("../t10k-labels.idx1-ubyte","rb");
    
    uint image_header[4];
    uint label_header[2];    
    fread(&image_header[0],sizeof(uint),4,test_image);
    fread(&label_header[0],sizeof(uint),2,test_label);

    uchar image_data[L1];
    uchar label_data;
    int COUNT=0;
    for(int var=0;var<test_count;var++)
    {
        fread(&image_data[0],sizeof(uchar),L1,test_image);
        for(int i=0;i<L1;i++)
        {
            input_layer[i]=((float)image_data[i])/255.0;
        }
        inference();
        int location=find_max();
        fread(&label_data,sizeof(uchar),1,test_label);
        if(location==label_data)COUNT++;
    }
    // printf("total:%d\ncorrect:%d\nrate:%f\n",test_count,COUNT,(float)COUNT/(test_count*1.0));
    cout<<"correct:"<<COUNT<<endl;
}
