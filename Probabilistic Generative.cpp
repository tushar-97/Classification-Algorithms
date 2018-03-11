#include <iostream>
#include <armadillo>
#include <iomanip>

using namespace std;
using namespace arma;


double sigmoid (double x){
    return (double) 1 / (1 + exp(-x));
}


void printResults(double tp, double fp, double tn, double fn) {
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

    cout<<setw(15)<<"Accuracy : "<<accuracy<<endl; cout<<"\n";
    cout<<setw(15)<<"Precision : "<<precision<<endl; cout<<"\n";
    cout<<setw(15)<<"Recall : "<<recall<<endl; cout<<"\n";
    cout<<setw(15)<<"F-Measure : "<<f_measure<<endl; cout<<"\n";
}


int main() {
    mat data, test_data, ones(500, 5), zeroes(600, 5);
    data.load("train.txt");
    test_data.load("test.txt");

    int a = 0, b = 0;
    for(int i = 0 ; i < data.n_rows ; i++) {
        if(data(i, 4) == 1) {
            for(int j = 0 ; j < 5 ; j++) {
                ones(a,j) = data(i,j);
            }
            a++;
        }
        else {
            for(int j = 0 ; j < 5 ; j++) {
                zeroes(b,j) = data(i,j);
            }
            b++;
        }
    }
    ones = ones.submat(0, 0, a - 1, 3);
    zeroes = zeroes.submat(0, 0, b - 1, 3);

    mat mean1;
    mat mean2;

    mean1 = mean(zeroes);
    mean2 = mean(ones);

    mean1 = mean1.t();
    mean2 = mean2.t();

    mat s1(4, 4);
    for(int i = 0 ; i < zeroes.n_rows ; i++) {
        mat xi = zeroes.submat(i, 0, i, 3);
        xi = xi.t();
        mat c1 = xi - mean1;
        s1 = s1 + (c1 * c1.t());
    }

    mat s2(4, 4);
    for(int i = 0 ; i < ones.n_rows ; i++) {
        mat xi = ones.submat(i, 0, i, 3);
        xi = xi.t();
        mat c2 = xi - mean2;
        s2 = s2 + (c2 * c2.t());
    }

    mat Sigma = s1 + s2;
    Sigma = Sigma / (double)data.n_rows;
    Sigma = inv(Sigma);

    mat w = Sigma * (mean1 - mean2);

    mat w0 = (-(double)1/2) * mean1.t() * Sigma * mean1;
    mat w1 = ((double)1/2) * mean2.t() * Sigma * mean2;

    double w0_value = w0(0, 0) + w1(0, 0) + log((double)zeroes.n_rows / (double)ones.n_rows);

    double tp = 0, tn = 0, fp = 0, fn = 0;
    for(int i = 0 ; i < test_data.n_rows ; i++) {
        mat xi = test_data.submat(i, 0, i, test_data.n_cols - 2);
        xi = xi.t();
        mat p = w.t() * xi;
        double y = p(0, 0);
        double px = y + w0_value;
        px = sigmoid(px);
        if(px > 0.5) {
            if(test_data(i, 4) == 0)
                tp++;
            else
                fp++;
        }
        else{
            if(test_data(i, 4) == 1)
                tn++;
            else
                fn++;
        }
    }

    printResults(tp, fp, tn, fn);

    return 0;
}
