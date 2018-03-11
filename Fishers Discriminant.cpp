#include <fstream>
#include <iostream>
#include <armadillo>
#include <iomanip>

using namespace std;
using namespace arma;


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

    mat Sw = s1 + s2;

    mat w = inv(Sw) * (mean2 - mean1);

    mat data_concise = data.submat(0, 0, data.n_rows - 1, 3);
    data_concise = data_concise.t();

    mat x_w = w.t() * data_concise;

    mat sorted_x_w = sort(x_w, "descend", 1);

    double min_entropy = 999999999;
    double threshold = 0;
    for(int i = 0 ; i < sorted_x_w.n_cols - 1 ; i++) {
        double fi = (sorted_x_w(0, i) + sorted_x_w(0, i + 1)) / 2;

        int left_plus = 0, left_minus = 0;
        int right_plus = 0, right_minus = 0;

        for(int j = 0 ; j < ones.n_rows ; j++) {
            mat xi = ones.submat(j, 0, j, 3);
            xi = xi.t();
            mat f = w.t() * xi;
            if(f(0, 0) >= fi)
                right_plus++;
            else
                left_plus++;

        }

        for(int j = 0 ; j < zeroes.n_rows ; j++) {
            mat xi = zeroes.submat(j, 0, j, 3);
            xi = xi.t();
            mat f = w.t() * xi;
            if(f(0, 0) < fi)
                left_minus++;
            else
                right_minus++;
        }

        double p_plus_left = (double)left_plus / (double)(left_plus + left_minus);
        double p_minus_left = (double)left_minus / (double)(left_plus + left_minus);
        double p_plus_right= (double)right_plus / (double)(right_plus + right_minus);
        double p_minus_right= (double)right_minus / (double)(right_plus + right_minus);

        double left_entropy = 0 - (p_plus_left * log2(p_plus_left)) - (p_minus_left * log2(p_minus_left));
        double right_entropy = 0 - (p_plus_right * log2(p_plus_right)) - (p_minus_right * log2(p_minus_right));
        double entropy = ((double)(left_plus + left_minus) * left_entropy + (double)(right_plus + right_minus) * right_entropy) / (double) (left_plus + left_minus + right_plus + right_minus);

        if(entropy < min_entropy) {
            min_entropy = entropy;
            threshold = fi;
        }
    }

    double tp = 0, tn = 0, fp = 0, fn = 0;
    for(int i = 0 ; i < test_data.n_rows ; i++) {
        mat xi = test_data.submat(i, 0, i, 3);
        xi = xi.t();
        mat xw = w.t() * xi;
        if(xw(0, 0) >= threshold) {
            if((int)test_data(i, 4) == 1)
                tp++;
            else
                fp++;
        }
        else {
            if((int)test_data(i, 4) == 0)
                tn++;
            else
                fn++;
        }
    }

	printResults(tp, fp, tn, fn);

    return 0;
}
