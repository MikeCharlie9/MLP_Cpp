#include "NNclass.cpp"
#include <windows.h>

void getTime()
{
    SYSTEMTIME sys; 
    GetLocalTime( &sys );
    printf("Local Time:%dmin %ds %dms\n", sys.wMinute, sys.wSecond,sys.wMilliseconds);
}

int main()
{
    NN mnist;
    mnist.init();
    int epoch=1;
    cout<<"start training"<<endl;
    getTime();
    for(int var=0;var<10;var++)
    {
        mnist.start_train(epoch);
        cout<<"epoch: "<<var<<" has been trained! ";
        // cout<<"start testing"<<endl;
        mnist.start_test();
    }    
    getTime();
    system("pause");
    return 0;
}