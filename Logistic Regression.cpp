#include <iostream>
#include <armadillo>
#include <iomanip>

using namespace std;
using namespace arma;

double sigmoid (double x){
    return (double) 1 / (1 + exp(-x));
}

void printResults(double tp, double fp, double tn, double fn, int epoch) {
    double accuracy = (tp + tn) / (tp + tn + fp + fn);
    double precision = (tp) / (tp + fp);
    double recall = (tp) / (tp + fn);
    double f_measure = 2 * (precision * recall) / (precision + recall);

    cout<<"\n\n";

    cout<<setw(15)<<"";
    cout<<setw(15)<<"Actual Yes";
    cout<<setw(15)<<"Actual No"<<endl;

    cout<<"\n";

    cout<<setw(15)<<"Predicted Yes";
    cout<<setw(15)<<tp;
    cout<<setw(15)<<fp<<endl;

    cout<<"\n";

    cout<<setw(15)<<"Predicted No";
    cout<<setw(15)<<fn;
    cout<<setw(15)<<tn<<endl;

    cout<<"\n\n";

    cout<<setw(15)<<"Iterations : "<<epoch<<endl; cout<<"\n";
    cout<<setw(15)<<"Accuracy : "<<accuracy<<endl; cout<<"\n";
    cout<<setw(15)<<"Precision : "<<precision<<endl; cout<<"\n";
    cout<<setw(15)<<"Recall : "<<recall<<endl; cout<<"\n";
    cout<<setw(15)<<"F-Measure : "<<f_measure<<endl; cout<<"\n";
}

int main() {
    mat data, test_data, ones(500, 5), zeroes(600, 5);
    data.load("train.txt");
    test_data.load("test.txt");
	double learning_rate;
	cout<<"Enter learning rate : ";
	cin>>learning_rate;

    mat col_test_data = test_data.t();
    mat t_test = col_test_data.submat(col_test_data.n_rows - 1, 0, col_test_data.n_rows - 1, col_test_data.n_cols - 1);
    for(int i = 0 ; i < col_test_data.n_cols ; i++)
        col_test_data(col_test_data.n_rows - 1, i) = 1;

    mat col_data = data.t();
    mat t = col_data.submat(col_data.n_rows - 1, 0, col_data.n_rows - 1, col_data.n_cols - 1);
    for(int i = 0 ; i < col_data.n_cols ; i++)
        col_data(col_data.n_rows - 1, i) = 1;

    mat w(5, 1);
    w.fill(1.0000);

    double e = learning_rate;
    double epsilon = 0.001;
    double stop_check;
    int epoch = 0;
    while(true) {
        mat y_matrix = w.t() * col_data;

        for(int i = 0 ; i < y_matrix.n_cols ; i++) {
            y_matrix(0, i) = sigmoid(y_matrix(0, i));
        }
        mat gradient = (y_matrix - t) * col_data.t();
        mat change = gradient.t();
        stop_check = -999999;
        for(int j = 0 ; j < w.n_rows ; j++) {
            if(stop_check < (abs(change(j, 0) / w(j, 0))))
                stop_check = abs(change(j, 0) / w(j, 0));
        }
        w = w - e * change;
        if(stop_check < epsilon) {
            break;
        }
        epoch++;
        if(epoch >= 100000) {
            break;
        }
    }

    mat y_matrix_test = w.t() * col_test_data;
    for(int i = 0 ; i < y_matrix_test.n_cols ; i++) {
        y_matrix_test(0, i) = sigmoid(y_matrix_test(0, i));
    }

    double tp = 0, tn = 0, fp = 0, fn = 0;
    for(int i = 0 ; i < y_matrix_test.n_cols ; i++) {
        if(y_matrix_test(0, i) >= 0.5) {
            if(t_test(0, i) == 1)
                tp++;
            else
                fp++;
        }
        else {
            if(t_test(0, i) == 0)
                tn++;
            else
                fn++;
        }
    }

    printResults(tp, fp, tn, fn, epoch);

    return 0;
}
