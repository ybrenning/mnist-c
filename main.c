#include <assert.h>
#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TRAIN_SIZE 60000
#define TEST_SIZE 10000

#define HEIGHT 28
#define WIDTH 28

#define MAXCHAR 5120

// TODO: Move struct declarations to header file
typedef struct Matrix {
    float **elems;
    int rows;
    int cols;
} matrix_t;

matrix_t *create_matrix(int rows, int cols)
{
    matrix_t *matrix = (matrix_t *) malloc(sizeof(matrix_t));
    if (matrix == NULL) {
        printf("Error while allocating matrix memory\n");
        return NULL;
    }
    matrix->rows = rows;
    matrix->cols = cols;

    matrix->elems = (float **) malloc(sizeof(float *) * rows);
    if (matrix->elems == NULL) {
        printf("Error while allocating matrix rows\n");
        return NULL;
    }
    for (int i = 0; i < rows; i++) {
        matrix->elems[i] = malloc(sizeof(float) * cols);
        if (matrix->elems[i] == NULL) {
            printf("Error while allocating matrix cols\n");
            return 0;
        }
    }

    return matrix;
}

void free_matrix(matrix_t *matrix)
{
    for (int i = 0; i < matrix->rows; i++) {
        free(matrix->elems[i]);
    }

    free(matrix->elems);
    free(matrix);
}

void print_matrix(matrix_t *matrix)
{
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            if (matrix->elems[i][j]) {
                printf("%.4f  ", matrix->elems[i][j]);
            } else {
                printf("0.  ");
            }
        }
        printf("\n");
    }
    printf("\n");
}

matrix_t *mul(matrix_t *a, matrix_t *b)
{
    if (a->cols != b->rows) {
        printf("Dimension error, matrices must be of size lxm * mxn");
        return NULL;
    }

    matrix_t *res = create_matrix(a->rows, b->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            float sum = 0.f;
            for (int k = 0; k < a->cols; k++) {
                sum += a->elems[i][k] * b->elems[k][j];
            }
            res->elems[i][j] = sum;
        }
    }

    return res;
}

matrix_t *mul_elems(matrix_t *a, matrix_t *b)
{
    if (a->rows != b->rows || a->cols != b->cols) {
        printf("Dimension mismatch, both matrices must be mxn\n");
        return NULL;
    }

    matrix_t *res = create_matrix(a->rows, b->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            res->elems[i][j] = a->elems[i][j] * b->elems[i][j];
        }
    }

    return res;
}

matrix_t *sum(matrix_t *a, matrix_t *b)
{
    if (a->rows != b->rows || a->cols != b->cols) {
        printf("Dimension mismatch, both matrices must be mxn\n");
        return NULL;
    }

    matrix_t *res = create_matrix(a->rows, a->cols);
    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < a->cols; j++) {
            res->elems[i][j] = a->elems[i][j] + b->elems[i][j];
        }
    }

    return res;
}

matrix_t *sub(matrix_t *a, matrix_t *b)
{
    if (a->rows != b->rows || a->cols != b->cols) {
        printf("Dimension mismatch, both matrices must be mxn\n");
        return NULL;
    }

    matrix_t *res = create_matrix(a->rows, a->cols);
    for (int i = 0; i < a->cols; i++) {
        for (int j = 0; j < a->cols; j++) {
            res->elems[i][j] = a->elems[i][j] - b->elems[i][j];
        }
    }

    return res;
}

matrix_t *fill(matrix_t *matrix, float num)
{
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            matrix->elems[i][j] = num;
        }
    }

    return matrix;
}

matrix_t *fill_rand(matrix_t *matrix)
{
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            matrix->elems[i][j] = (float) rand() / RAND_MAX;
        }
    }

    return matrix;
}

typedef struct Image {
    matrix_t *content;
    int label;
} img;

img **read_csv(const char *filename, int num_samples)
{
    img **images = (img **) malloc(sizeof(img *) * num_samples);
    if (images == NULL) {
        printf("Error while allocating dataset memory\n");
        return NULL;
    }

    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        printf("File %s not found\n", filename);
        return NULL;
    }

    char row[MAXCHAR];
    // Skip first line of csv
    fgets(row, MAXCHAR, fp);
    char *token = strtok(row, ",");

    int current_row = 0;
    while (!feof(fp) && current_row < num_samples) {
        int x = 0, y = 0;

        images[current_row] = (img *) malloc(sizeof(img));
        if (images[current_row] == NULL) {
            printf("Error while allocating memory for image no. %d\n", current_row);
            fclose(fp);
            return NULL;
        }

        fgets(row, MAXCHAR, fp);
        token = strtok(row, ",");

        images[current_row]->label = atoi(token);
        token = strtok(NULL, ",");
        images[current_row]->content = create_matrix(HEIGHT, WIDTH);

        // Parsing row of csv
        while (token) {
            // This is super ugly but it works for now
            if (y == 28) {
                y = 0;
                x++;
            }
            if (x == 28) {
                break;
            }

            images[current_row]->content->elems[x][y++] = (float) atoi(token) / 255.;
            token = strtok(NULL, ",");
        }

        current_row++;
    }

    fclose(fp);
    return images;
}

void free_img(img *image)
{
    free_matrix(image->content);
    free(image);
}

void free_imgs(img **images, int num_samples)
{
    for (int i = 0; i < num_samples; i++) {
        free_img(images[i]);
    }

    free(images);
}

void print_img(img *image)
{
    printf("Label: %d\n\n", image->label);
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            printf("%.2f\t", image->content->elems[i][j]);
        }

        printf("\n");
    }
}

float sigmoidf(float x)
{
    return 1.f / (1 + expf(-x));
}

float d_sigmoidf(float x)
{
    return x * (1.f - x);
}

matrix_t *matrix_sigmoid(matrix_t *matrix)
{
    matrix_t *res = create_matrix(matrix->rows, matrix->cols);
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            res->elems[i][j] = sigmoidf(matrix->elems[i][j]);
        }
    }

    return res;
}

matrix_t *matrix_d_sigmoid(matrix_t *matrix)
{
    matrix_t *res = create_matrix(matrix->rows, matrix->cols);
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            res->elems[i][j] = d_sigmoidf(matrix->elems[i][j]);
        }
    }

    return res;
}

matrix_t *softmax(matrix_t *matrix)
{
    matrix_t *res = create_matrix(matrix->rows, matrix->cols);
    float sum = 0.f;
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            sum += expf(matrix->elems[i][j]);
        }
    }

    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            res->elems[i][j] = expf(matrix->elems[i][j]) / sum;
        }
    }

    return res;
}

matrix_t *transpose(matrix_t *matrix)
{
    matrix_t *res = create_matrix(matrix->cols, matrix->rows);
    for (int i = 0; i < matrix->rows; i++) {
        for (int j = 0; j < matrix->cols; j++) {
            res->elems[j][i] = matrix->elems[i][j];
        }
    }

    return res;
}

matrix_t *flatten(matrix_t *matrix, int axis)
{
    if (axis == 0) {
        matrix_t *res = create_matrix(matrix->rows * matrix->cols, 1);
        for (int i = 0; i < matrix->rows; i++) {
            for (int j = 0; j < matrix->cols; j++) {
                res->elems[i * matrix->cols + j][0] = matrix->elems[i][j];
            }
        }
        return res;
    } else if (axis == 1) {
        matrix_t *res = create_matrix(1, matrix->rows * matrix->cols);
        for (int i = 0; i < matrix->rows; i++) {
            for (int j = 0; j < matrix->cols; j++) {
                res->elems[0][i * matrix->cols + j] = matrix->elems[i][j];
            }
        }
        return res;
    } 

    assert(0 && "Wrong axis");
    return NULL;
}

typedef struct NeuralNetwork {
    int input;
    int hidden;
    int output;
    float lr;
    matrix_t *hidden_w;
    matrix_t *output_w;
} nn;

nn *create_nn(int input, int hidden, int output, float lr)
{
    nn *nn = malloc(sizeof(struct NeuralNetwork));
    nn->input = input;
    nn->hidden = hidden;
    nn->output = output;
    nn->lr = lr;
    printf("Creating hidden weights (%d, %d)\n", hidden, input);
    printf("Creating output weights (%d, %d)\n", output, hidden);
    nn->hidden_w = create_matrix(hidden, input);
    nn->output_w = create_matrix(output, hidden);
    nn->hidden_w = fill_rand(nn->hidden_w);
    nn->output_w = fill_rand(nn->output_w);
    return nn;
}

void free_nn(nn *nn)
{
    free_matrix(nn->hidden_w);
    free_matrix(nn->output_w);
    free(nn);
}

matrix_t *scale(matrix_t *m, float s)
{
    matrix_t *res = create_matrix(m->rows, m->cols);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            res->elems[i][j] = m->elems[i][j] * s;
        }
    }

    return res;
}

matrix_t *forward(nn *nn, matrix_t *x) {
    matrix_t *hidden_activations = matrix_sigmoid(mul(nn->hidden_w, x));
    matrix_t *output_activations = matrix_sigmoid(mul(nn->output_w, hidden_activations));
    return output_activations;
}

void train_nn(nn *nn, img **imgs, int *indices, int batch_start, int batch_end)
{
    // print_matrix(nn->hidden_w);
    // print_matrix(X_train);
    // print_matrix(mul(nn->hidden_w, X_train));
    // print_matrix(matrix_sigmoid(mul(nn->hidden_w, X_train)));
    for (int i = batch_start; i < batch_end; i++) {
        int j = indices[i];
        img *current_image = imgs[j];
        matrix_t *x = flatten(current_image->content, 0);
        matrix_t *y = create_matrix(10, 1);
        y->elems[current_image->label][0] = 1.f;

        free_matrix(x);
        free_matrix(y);
    }
    // matrix_t *hidden_outputs = matrix_sigmoid(mul(nn->hidden_w, X_train));
    // matrix_t *final_outputs = matrix_sigmoid(mul(nn->output_w, hidden_outputs));
    //
    // // Error
    // matrix_t *output_e = sub(y_train, final_outputs);
    // matrix_t *hidden_e = mul(transpose(nn->output_w), output_e);
    //
    // // Backprop
    // matrix_t *d_sig_mat = matrix_d_sigmoid(final_outputs);
    // matrix_t *mul_mat = mul(output_e, d_sig_mat);
    // matrix_t *transposed_mat = transpose(hidden_outputs);
    // matrix_t *dot_mat = mul(mul_mat, transposed_mat);
    // matrix_t *scaled_mat = scale(dot_mat, nn->lr);
    // matrix_t *added_mat = sum(nn->output_w, scaled_mat);
    //
    // free_matrix(nn->output_w);
    // nn->output_w = added_mat;
    //
    // free_matrix(d_sig_mat);
    // free_matrix(mul_mat);
    // free_matrix(transposed_mat);
    // free_matrix(dot_mat);
    // free_matrix(scaled_mat);
    //
    // d_sig_mat = matrix_d_sigmoid(hidden_outputs);
    // mul_mat = mul(hidden_e, d_sig_mat);
    // transposed_mat = transpose(X_train);
    // dot_mat = mul(mul_mat, transposed_mat);
    // scaled_mat = scale(dot_mat, nn->lr);
    // added_mat = sum(nn->hidden_w, scaled_mat);
    //
    // free_matrix(nn->hidden_w);
    // nn->hidden_w = added_mat;
    //
    // free_matrix(d_sig_mat);
    // free_matrix(mul_mat);
    // free_matrix(transposed_mat);
    // free_matrix(dot_mat);
    // free_matrix(scaled_mat);
    //
    // free_matrix(hidden_outputs);
    // free_matrix(final_outputs);
    // free_matrix(hidden_e);
    // free_matrix(output_e);
}

void swap(int *a, int *b)
{
    int temp = *a;
    *a = *b;
    *b = temp;
}
 
void shuffle(int arr[], int n)
{
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i+1);
        swap(&arr[i], &arr[j]);
    }
}

void train_batch_nn(nn *nn, img **imgs, int batch_size, int n_epochs)
{
    int n_batches = TRAIN_SIZE / batch_size;
    int *indices = (int *) malloc(sizeof(int) * TRAIN_SIZE);
    for (int i = 0; i < TRAIN_SIZE; i++) {
        indices[i] = i;
    }
    shuffle(indices, TRAIN_SIZE);

    for (int i = 0; i < n_epochs; i++) {
        for (int j = 0; j < n_batches - 1; j++) {
            int batch_start = j*batch_size;
            int batch_end = j*batch_size + batch_size;
            train_nn(nn, imgs, indices, batch_start, batch_end);

        }
    }
}

matrix_t *nn_predict(nn *nn, matrix_t *X)
{
    matrix_t *hidden_outputs = matrix_sigmoid(mul(nn->hidden_w, X));
    matrix_t *final_outputs = matrix_sigmoid(mul(nn->output_w, hidden_outputs));
    matrix_t *res = softmax(final_outputs);

    free_matrix(hidden_outputs);
    free_matrix(final_outputs);
    return res;
}

matrix_t *predict_img(nn *nn, img *img)
{
    matrix_t *img_data = flatten(img->content, 0);
    matrix_t *res = nn_predict(nn, img_data);
    free_matrix(img_data);
    return res;
}

int argmax(matrix_t *matrix)
{
    int max = 0;
    for (int i = 0; i < matrix->cols; i++) {
        if (matrix->elems[0][i] >= matrix->elems[0][0]) {
             max = i;
        }
    }

    return max;
}

double predict_imgs(nn *nn, img **imgs, int num_samples)
{
    int correct = 0;
    for (int i = 0; i < num_samples; i++) {
        matrix_t *pred = predict_img(nn, imgs[i]);
        if (argmax(pred) == imgs[i]->label) {
            correct++;
        }
        free_matrix(pred);
    }

    return 1.f * correct / num_samples;
}

int main(void)
{
    printf("Don't forget to activate random seed with srand()\n");

    img **X_train = read_csv("data/mnist_train.csv", TRAIN_SIZE / 2);
    img **X_test = read_csv("data/mnist_test.csv", TEST_SIZE / 2);

    // Print first sample
    print_img(X_train[0]);

    nn *nn = create_nn(784, 128, 10, 0.01);
    train_batch_nn(nn, X_train, 100, 10);
    // printf("Score %.4f\n", predict_imgs(nn, X_test, TEST_SIZE / 4));
    //
    // free_nn(nn);

    free_imgs(X_train, TRAIN_SIZE / 2);
    free_imgs(X_test, TEST_SIZE / 2);

    return 0;
}
