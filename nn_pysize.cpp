#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <vector>
#include <string>
#include <omp.h>
#include <cmath>

Eigen::MatrixXd xavierInit(Eigen::MatrixXd &weights, int inputSize, int outputSize) {
    double limit = std::sqrt(6.0 / (inputSize + outputSize));
    weights = Eigen::MatrixXd::Random(inputSize, outputSize) * limit;
    return weights;
}


Eigen::MatrixXd matrix_creation(std::string path){
    std::ifstream x_train_file(path);
    std::string line;
    std::vector<std::vector<float>> matrix_2; 
    std::vector<float> matrx_2; 
    std::string val = "";
    int itter = 0;
    int com_itter = 0;
        

    while (std::getline(x_train_file, line)) {
        
        if (line[line.length() - 1] == ','){
            line = "," + line;
        } else {
            line = "," + line + ",";
        }
        
        com_itter = 0;
        itter = 0;

        for (char c : line){
            if (c == ','){
                com_itter += 1;
            }

            if (com_itter == 1 && c != ','){
                val += c;
            }

            if (com_itter == 2){
                matrx_2.push_back(std::stof(val));
                com_itter = 1;
                val = "";
            }

            itter += 1;

            if (itter == line.length()){
                matrix_2.push_back(matrx_2);
                matrx_2.clear();
            }
        }
                
    }

    //CONVERT INTO A MATRIX
    int rows = matrix_2.size();
    int cols = matrix_2[0].size();
    
    Eigen::MatrixXd matrix(rows, cols);
    for (int r = 0; r < rows; r++){
        for (int c = 0; c < cols; c++){
            matrix(r,c) = matrix_2[r][c];
        }
    }

    return matrix; 

}



std::vector<Eigen::MatrixXd> init_parameters_n_bias(){
    std::vector<Eigen::MatrixXd> parameters_n_bias;

    /*W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2*/

    Eigen::MatrixXd W1(10, 784);
    W1 = xavierInit(W1, 10,784);

    Eigen::MatrixXd B1(10, 10);
    B1 = xavierInit(B1, 10,1);

    Eigen::MatrixXd W2(10, 10);
    W2 = xavierInit(W2, 10,10);

    Eigen::MatrixXd B2(10, 10);
    B2 = xavierInit(B2, 10,1);


    parameters_n_bias.push_back(W1);
    parameters_n_bias.push_back(B1); 
    parameters_n_bias.push_back(W2); 
    parameters_n_bias.push_back(B2); 
  
    return parameters_n_bias;

}

Eigen::MatrixXd prelu(Eigen::MatrixXd out){
    return out.cwiseMax(0.0);
}


Eigen::MatrixXd softmax(Eigen::MatrixXd out){
    float sum = 0.f;
    Eigen::MatrixXd softout = out;
    Eigen::MatrixXd colly(softout.rows(), softout.cols());
    //std::cout << "output = " << neurons[neurons.size() - 1] << std::endl;
        
    for (int c = 0; c < softout.cols(); c++){
                
        for (int r = 0; r < out.rows(); r++){
            sum += std::exp(softout(r,c));
        }

        for (int r = 0; r < out.rows(); r++){
            colly(r,c) = std::exp(softout(r,c)) / sum;
        }
        
        sum = 0.f;
    }

    return colly;
}



std::vector<Eigen::MatrixXd> feed_forward(Eigen::MatrixXd W1, Eigen::MatrixXd B1, Eigen::MatrixXd W2, Eigen::MatrixXd B2, Eigen::MatrixXd X){

    std::vector<Eigen::MatrixXd> all_neurons;

    //std::cout << "W1(0,0) in feed forward =  " << W1(0,0) << std::endl;

    Eigen::MatrixXd Z1 = W1 * X + B1; 
    //std::cout << "Z1 = " << Z1.rows() << "," << Z1.cols() << std::endl;
    Eigen::MatrixXd A1 = prelu(Z1); 
    //std::cout << "A1 = " << A1.rows() << "," << A1.cols() << std::endl;
    Eigen::MatrixXd Z2 = W2 * A1 + B2;
    //std::cout << "Z2 = " << Z2.rows() << "," << Z2.cols() << std::endl;
    Eigen::MatrixXd A2 = softmax(Z2); 
    //std::cout << "A2 = " << A2.rows() << "," << A2.cols() << std::endl;
    //std::cout << "A2.col(0) = " << A2.col(0) << std::endl;
    
    all_neurons.insert(all_neurons.end(),{Z1,A1,Z2,A2});
    return all_neurons;

}



Eigen::MatrixXd vectorise_label_cols(Eigen::MatrixXd labels, Eigen::MatrixXd predicted){
    Eigen::MatrixXd base(predicted.rows(),predicted.cols());
    base.setZero();
    
    for (int c = 0; c < predicted.cols(); c++){
        int label = labels(0,c);
        base(label,c) = 1;
    }

    return base;

} 



std::vector<Eigen::MatrixXd> back_propagation(std::vector<Eigen::MatrixXd> neurons, Eigen::MatrixXd W2, Eigen::MatrixXd X, Eigen::MatrixXd y_train){
    std::vector<Eigen::MatrixXd> gradients; 

    
    /*one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2) #guy used no axis = 1
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1) 
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2*/
    
    //std::cout << "here";
    Eigen::MatrixXd DZ2 = neurons[neurons.size() - 1] - vectorise_label_cols(y_train, neurons[neurons.size() - 1]); //10 * 10000
    //std::cout << "DZ2 = " << DZ2.rows() << "," << DZ2.cols() << std::endl;
    Eigen::MatrixXd DW2 = (DZ2 * neurons[neurons.size() - 3].transpose()) * 1/10000; // 10 * 150
    //std::cout << "DW2 = " << DW2.rows() << "," << DW2.cols() << std::endl;
    double DB2_v = DZ2.sum() * 1/10000;
    //std::cout << "DB2_v = " << DB2_v << std::endl;
    Eigen::MatrixXd DB2 = Eigen::MatrixXd::Constant(10, X.cols(), DB2_v);//<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< double in python
    //std::cout << "DB2.col(0) = " << DB2.col(0) << std::endl;

    Eigen::MatrixXd DZ1_comp = W2.transpose() * DZ2;
    Eigen::MatrixXd DZ1 = DZ1_comp.array() * neurons[1].array();
    //std::cout << "neurons[1].col(0) = " << neurons[1].col(0) << "\n\n";
    //std::cout << "DZ1.col(0) = " << DZ1.col(0) << "\n\n";
    //std::cout << "DZ1.rows() = " << DZ1.rows() << "," << "DZ1.cols() = " << DZ1.cols() << std::endl;
    Eigen::MatrixXd DW1 = (DZ1 * X.transpose()) * 1/10000;
    //std::cout << "DW1.rows() = " << DW1.rows() << "," << "DW1.cols() = " << DW1.cols() << std::endl; 
    double DB1_v = DZ1.sum() * 1/10000;
    //std::cout << "DB1_v = " << DB1_v << std::endl; 
    Eigen::MatrixXd DB1 = Eigen::MatrixXd::Constant(10, X.cols(), DB1_v); //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< double in python
    //std::cout << "DB1.rows() = " << DB1.rows() << "," << "DB1.cols() = " << DB1.cols() << std::endl; 
    
    

    gradients.insert(gradients.end(), {DW1,DB1,DW2,DB2});
    return gradients;

}



std::vector<Eigen::MatrixXd> update_params(std::vector<Eigen::MatrixXd> gradients, Eigen::MatrixXd W1, Eigen::MatrixXd B1_x, Eigen::MatrixXd W2, Eigen::MatrixXd B2_x, float alpha){
    std::vector<Eigen::MatrixXd> up_par;
    
    Eigen::MatrixXd new_W1 = W1 - alpha * gradients[0];
    //std::cout << "gradients[0] = " << gradients[0] << std::endl;
    Eigen::MatrixXd new_B1 = B1_x - alpha * gradients[1];
    Eigen::MatrixXd new_W2 = W2 - alpha * gradients[2];
    Eigen::MatrixXd new_B2 = B2_x - alpha * gradients[3];

    up_par.insert(up_par.end(), {new_W1,new_B1,new_W2,new_B2});
    return up_par;
    
    //{DW1,DB1,DW2,DB2,DW3,DB3}

}


Eigen::MatrixXd argmax(Eigen::MatrixXd predicted){
    Eigen::MatrixXd base(1, predicted.cols());
    Eigen::Index maxRow, maxCol;

    for (int c = 0; c < predicted.cols(); c++){
        float max = predicted.col(c).maxCoeff(&maxRow, &maxCol);
        base(0,c) = maxRow;
    }

    return base;

} 

int main(){

    Eigen::MatrixXd x_t = matrix_creation("/home/kali/Desktop/machine_learning/Neural_networks/from_scratch/cpp_networks/x_train.csv");
    Eigen::MatrixXd y_t = matrix_creation("/home/kali/Desktop/machine_learning/Neural_networks/from_scratch/cpp_networks/y_train.csv");
    Eigen::MatrixXd x_train = x_t.block(0,0,8000,x_t.cols()).transpose()/255;
    Eigen::MatrixXd y_train = y_t.block(0,0,8000,y_t.cols()).transpose();
    
    std::cout << "x_train = " << x_train.rows() << "," << x_train.cols() << std::endl;
    std::cout << "y_train = " << y_train.rows() << "," << y_train.cols() << std::endl;

    std::vector<Eigen::MatrixXd> parameters_n_bias = init_parameters_n_bias();       
    Eigen::MatrixXd W1 = parameters_n_bias[0];
    //std::cout << "original W1 = " << W1.col(0) << "\n\n";
    Eigen::MatrixXd B1 = parameters_n_bias[1];
    Eigen::MatrixXd B1_x = B1.replicate(1, x_train.cols());
    Eigen::MatrixXd W2 = parameters_n_bias[2];
    Eigen::MatrixXd B2 = parameters_n_bias[3];
    Eigen::MatrixXd B2_x = B2.replicate(1, x_train.cols());

    float alpha = 0.1f;

    for (int ittt = 0; ittt < 5000; ittt++){
        std::vector<Eigen::MatrixXd> neurons = feed_forward(W1, B1_x, W2, B2_x, x_train);
        std::vector<Eigen::MatrixXd> gradients = back_propagation(neurons, W2, x_train, y_train);
        std::vector<Eigen::MatrixXd> updatedparams = update_params(gradients, W1, B1_x, W2, B2_x,alpha);

        B2_x = updatedparams[updatedparams.size() - 1];
        W2 = updatedparams[updatedparams.size() - 2];
        B1_x = updatedparams[updatedparams.size() - 3];
        W1 = updatedparams[updatedparams.size() - 4];

        //std::cout << "W1.col(0) = " << W1.col(0) << std::endl;

        if (ittt % 10 == 0 && ittt != 0){//ittt % 10 == 0 && ittt != 0
            Eigen::MatrixXd predicted = argmax(neurons[neurons.size() - 1]);
            //std::cout << "neurons[neurons.size() - 1] = " << neurons[neurons.size() - 1].col(0) << std::endl;
            //std::cout << "predicted.col(0) = " << predicted.col(0) << std::endl;
            int acc = (predicted.array() == y_train.array()).count();
            float accuracy = acc / 80.f;
            std::cout << "itter: " << ittt << "  accuracy:  " << accuracy << std::endl;            
           
        }

    }

    return 0;
}